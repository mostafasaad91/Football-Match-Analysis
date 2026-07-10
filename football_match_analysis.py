#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportRedeclaration=false, reportReturnType=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportPossiblyUnboundVariable=false
"""
WhoScored Post-Match Analyzer  ·  v7 internal xG engine  ·  2026-04-30
=======================================================
✅ Robust handling for WhoScored blocking
   ├─ Attempt 1: cloudscraper (no browser; fastest)
   ├─ Attempt 2: requests with rotating headers
   └─ Attempt 3: undetected_chromedriver with a real Chrome profile and stealth

✅ Dark theme across all 11 figures
✅ Own goals use the benefiting team's color
✅ xT map and full match report

Install the required packages:
  pip install cloudscraper undetected-chromedriver selenium scipy
              beautifulsoup4 numpy pandas matplotlib rich
"""

# ══════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════
import ast, json, math, os, re, sys, time, random, warnings, shutil, tempfile, hashlib
import numpy as np
import pandas as pd
import matplotlib

# ── Rendering mode ─────────────────────────────────────────────
# False = headless save-only mode (recommended when generating many figures/PDFs)
# True  = open interactive matplotlib windows after finishing
SHOW_WINDOWS = False
matplotlib.use("TkAgg" if SHOW_WINDOWS else "Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from bs4 import BeautifulSoup
from datetime import datetime
from rich.console import Console
from rich.table import Table
from matplotlib.backends.backend_pdf import PdfPages

os.environ["MATCH_ANALYSIS_THEME"] = "dark"
SOFASCORE_PLAYER_TABLES = False
SOFASCORE_AUTO_SEARCH = False
SOFASCORE_EVENT_ID = 15186710
SOFASCORE_MIN_MATCH_CONFIDENCE = 0.82
os.environ["SOFASCORE_PLAYER_TABLES"] = "1" if SOFASCORE_PLAYER_TABLES else "0"
os.environ["SOFASCORE_AUTO_SEARCH"] = "1" if SOFASCORE_AUTO_SEARCH else "0"
os.environ["SOFASCORE_MIN_MATCH_CONFIDENCE"] = str(SOFASCORE_MIN_MATCH_CONFIDENCE)
if SOFASCORE_EVENT_ID is not None:
    os.environ["SOFASCORE_EVENT_ID"] = str(SOFASCORE_EVENT_ID)
from match_report import run_analysis as _run_extended_analysis

# v2 redesigned visuals (xG flow, shot map, shot breakdown, pass network, xT map)
os.environ["MATCH_ANALYSIS_THEME"] = "dark"
try:
    from tactical_visualizations import (
        make_xg_flow_v2,
        make_shot_map_v2,
        make_shot_breakdown_v2,
        make_pass_network_v2,
        make_xt_map_v2,
    )

    _V2_AVAILABLE = True
except Exception:
    _V2_AVAILABLE = False

if _V2_AVAILABLE:
    import tactical_visualizations as _vc_runtime

    # ── Runtime self-check + hard override for the gk/def/mid/att colour
    # logic used by Pass Network and Average Positions. ────────────────
    # Background: this function must classify a player's role from the
    # ORIGINAL event coordinate system, where `x` is pitch depth
    # (0 = own goal line, 100 = opponent's goal line) and `y` is pitch
    # width. A previous version of this codebase read `y` instead of `x`,
    # which silently scrambled every node colour. We force the correct
    # implementation here, at the call site, so this report's colours are
    # correct even if some other copy of tactical_visualizations.py — e.g. one
    # picked up from a stale install, a different virtualenv, or any
    # other entry earlier on sys.path — would otherwise shadow the fixed
    # version on disk. This removes any dependency on cache state,
    # sys.path ordering, or which file a given Python environment resolves
    # the `tactical_visualizations` import to.
    def _infer_position_bucket_FORCED(
        p: dict, depth_axis: str = "x", direction: int = 1
    ) -> str:
        explicit = str(p.get("position") or p.get("pos") or "").strip().upper()
        if explicit:
            if explicit in {"GK", "GOALKEEPER"}:
                return "gk"
            if explicit.startswith(("D", "CB", "LB", "RB", "WB")):
                return "def"
            if explicit.startswith(("M", "DM", "CM", "AM")):
                return "mid"
            if explicit.startswith(("F", "FW", "ST", "CF", "W", "LW", "RW")):
                return "att"
        raw = p.get(depth_axis, 50)
        depth = raw if direction >= 0 else (100 - raw)
        if depth <= 8:
            return "gk"
        if depth <= 38:
            return "def"
        if depth <= 68:
            return "mid"
        return "att"

    _existing_src = ""
    try:
        import inspect as _inspect_check

        _existing_src = _inspect_check.getsource(_vc_runtime._infer_position_bucket)
    except Exception:
        pass

    _is_stale = ('p.get("y"' in _existing_src) or ("p.get('y'" in _existing_src)
    if _is_stale:
        print(
            "⚠️  WARNING: tactical_visualizations._infer_position_bucket was loaded from "
            f"{getattr(_vc_runtime, '__file__', '?')} using the OLD (y-as-depth) "
            "logic. Forcing the corrected (x-as-depth) version for this run. "
            "You should find and remove/update the stale copy of "
            "tactical_visualizations.py at that path so this override isn't needed."
        )
    else:
        print(
            f"✅ tactical_visualizations loaded from {getattr(_vc_runtime, '__file__', '?')} "
            "— bucket logic already correct (x-as-depth); applying override "
            "anyway as a guarantee."
        )

    _vc_runtime._infer_position_bucket = _infer_position_bucket_FORCED

warnings.filterwarnings("ignore")
console = Console()


# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
MATCH_URL = "https://www.whoscored.com/matches/1998582/live/international-fifa-world-cup-2026-france-morocco"
SAVE_DIR = "output"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(SAVE_DIR):
    SAVE_DIR = os.path.join(SCRIPT_DIR, SAVE_DIR)


def _safe_output_name(value: str, max_len: int = 90) -> str:
    value = re.sub(r"[^\w\s.-]+", "", str(value or ""), flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value.strip())
    value = value.strip("._-")
    return (value or "match")[:max_len]


def _match_output_folder(info: dict, root: str = SAVE_DIR) -> str:
    home = info.get("home_name") or "Home"
    away = info.get("away_name") or "Away"
    score = ""
    score_text = str(info.get("score") or "").strip()
    if score_text:
        score = "_" + re.sub(r"\s+", "", score_text.replace(":", "-"))
    elif info.get("home_score") is not None and info.get("away_score") is not None:
        score = f"_{info.get('home_score')}-{info.get('away_score')}"
    return os.path.join(root, _safe_output_name(f"{home}_vs_{away}{score}"))


# Kit colour mode used by every visual and PDF page.
# Options:
#   "auto"   -> current smart choice with clash/readability protection
#   "home"   -> first colour in the team's kit palette
#   "accent" -> second colour/accent in the team's kit palette
#   "away" / "third" / "alternate" -> third colour in the team's kit palette
# You can also force exact match-day kit colours with CUSTOM_KIT_COLORS below.
HOME_KIT_TYPE = "home"
AWAY_KIT_TYPE = "auto"
CUSTOM_KIT_COLORS = {
    # "home": "#C8102E",
    # "away": "#034694",
}

CHROMEDRIVER_PATH = ""


#   (Get-Item "$env:LOCALAPPDATA\Google\Chrome\User Data").FullName
CHROME_PROFILE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data"
)
if not CHROME_PROFILE_DIR or not os.path.isdir(CHROME_PROFILE_DIR):
    CHROME_PROFILE_DIR = r"C:\Users\Mostafa.saad\AppData\Local\Google\Chrome\User Data"
CHROME_PROFILE_NAME = "Default"


BROWSER_USE_REAL_PROFILE = False
BROWSER_HEADLESS = True
BROWSER_DOM_FALLBACK_ENABLED = True


LAST_PAGE_HTML = ""
LAST_PAGE_TEXT = ""


OFFICIAL_REPORT_OVERRIDE = {"enabled": False}


STRICT_OFFICIAL_PAGE_XG = False

# xG behaviour — V7 INTERNAL ENGINE
# No manually pasted public-site totals and no external xG matching.
# The model computes xG inside the script from the event data and match statistics
# available in WhoScored/matchCentreData: shot location, angle, distance, body part,
# qualifiers, big chances, penalties, direct free kicks, rebounds, cut-backs, through
# balls, crosses, set pieces, shot volume, shots on target and woodwork.
#
# Priority order:
#   1) assign shot-level xG with the internal event-context model;
#   2) estimate a team-level target from available match statistics;
#   3) bounded-rescale each team's shot values to that internal target.
XG_USE_PROVIDER_SHOT_XG = (
    False  # keep fully internal; ignore embedded provider xG if present
)
XG_USE_OFFICIAL_TEAM_TOTAL_CALIBRATION = (
    False  # do NOT calibrate to WhoScored/Opta/provider team xG totals
)
XG_USE_INTERNAL_TEAM_STAT_CALIBRATION = False
XG_LOCAL_MODEL_VERSION = "xg_v3_submodel_calibrated_2025"
XG_SINGLE_SHOT_CAP = 0.95
XG_PENALTY_VALUE = 0.79

# Internal team-stat target blend.
XG_INTERNAL_TEAM_PRIOR_WEIGHT = 0.00
XG_INTERNAL_TARGET_MULTIPLIER_MIN = 0.72
XG_INTERNAL_TARGET_MULTIPLIER_MAX = 1.38
XG_INTERNAL_BIG_CHANCE_VALUE = 0.29
XG_INTERNAL_ON_TARGET_BONUS = 0.024
XG_INTERNAL_WOODWORK_BONUS = 0.038

# Higher-resolution export settings.
# The first eight figures used to be saved at 150 DPI; all visuals now use the
# same high-resolution setting, and grouped boards render embedded figures at a
# higher internal DPI before saving.
OUTPUT_IMAGE_DPI = 420
PDF_EXPORT_DPI = 400
# Boards re-render every source figure into an RGBA buffer at this DPI, then
# tile 10 of them onto one large canvas. With ~40 figures already held open,
# 300 DPI here exhausts RAM on the biggest boards (MemoryError). 170 keeps the
# collages sharp while cutting the peak buffer memory to roughly a third.
BOARD_RENDER_DPI = 170
BOARD_SAVE_DPI = 360
GROUP_BOARD_MAX_VISUALS = 6


# ══════════════════════════════════════════════════════
#  COLORS & CONSTANTS
# ══════════════════════════════════════════════════════
C_BLUE = "#4D8DFF"
C_RED = "#FF4D4D"
C_GREEN = "#22c55e"
C_GOLD = "#FFC23C"

# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
TEAM_COLORS = {
    "Arsenal": "#EF0107",
    "Manchester City": "#6CABDD",
    "Manchester United": "#DA291C",
    "Liverpool": "#C8102E",
    "Chelsea": "#034694",
    "Tottenham": "#FFFFFF",
    "Newcastle": "#2D2D2D",  # Black/White stripes (visible dark charcoal for dark charts)
    "Aston Villa": "#95BFE5",
    "West Ham": "#7A263A",
    "Brighton": "#0057B8",
    "Brentford": "#E30613",
    "Fulham": "#000000",
    "Crystal Palace": "#1B458F",
    "Wolves": "#FDB913",
    "Everton": "#003399",
    "Nottm Forest": "#DD0000",
    "Bournemouth": "#DA291C",
    "Leicester": "#003090",
    "Ipswich": "#0044A9",
    "Southampton": "#D71920",
    "Barcelona": "#A50044",
    "Real Madrid": "#FEBE10",
    "Atletico Madrid": "#CB3524",
    "Bayern Munich": "#DC052D",
    "Borussia Dortmund": "#FDE100",
    "Juventus": "#000000",
    "Inter Milan": "#010E80",
    "AC Milan": "#FB090B",
    "PSG": "#004170",
}


TEAM_ALIASES = {
    "man city": "Manchester City",
    "man. city": "Manchester City",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "man. united": "Manchester United",
    "spurs": "Tottenham",
    "newcastle united": "Newcastle",
    "newcastle utd": "Newcastle",
    "newcastle": "Newcastle",
    "tottenham hotspur": "Tottenham",
    "wolverhampton": "Wolves",
    "nottingham forest": "Nottm Forest",
    "forest": "Nottm Forest",
    "west ham united": "West Ham",
    "brighton & hove albion": "Brighton",
    "leicester city": "Leicester",
    "ipswich town": "Ipswich",
    "atletico": "Atletico Madrid",
    "atlético madrid": "Atletico Madrid",
    "fc barcelona": "Barcelona",
    "bayern": "Bayern Munich",
    "fc bayern": "Bayern Munich",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "milan": "AC Milan",
    "paris saint-germain": "PSG",
    "paris sg": "PSG",
    # International team name variants (WhoScored/Opta sometimes use the
    # endonym or a different romanization than the palette key below).
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "republic of turkiye": "Turkey",
    "south korea": "Korea Republic",
    "korea republic": "Korea Republic",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d’ivoire": "Côte d'Ivoire",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "usa": "United States",
    "us soccer": "United States",
}

# ══════════════════════════════════════════════════════
#  TOP-5 LEAGUES 2025/26 — KIT-BASED VISUAL PALETTES
# ══════════════════════════════════════════════════════
# Format: canonical team name -> [home-kit dominant colour, accent/stripe, alternate clash colour]
# The first colour follows the shirt identity; the remaining colours are used automatically
# when two teams have very similar visual colours on the same chart.
TOP5_2025_26_TEAM_PALETTES = {
    # Premier League 2025/26
    "Arsenal": ["#EF0107", "#FFFFFF", "#063672"],  # Home: Red | Away: Navy Blue
    "Aston Villa": ["#7A003C", "#95BFE5", "#FEE505"],  # Home: Claret | Away: Light Blue
    "Bournemouth": ["#DA291C", "#000000", "#F7C600"],  # Home: Red | Away: Black
    "Brentford": ["#E30613", "#FFFFFF", "#111111"],  # Home: Red | Away: Black
    "Brighton": ["#0057B8", "#FFFFFF", "#FFCD00"],  # Home: Blue | Away: Yellow
    "Burnley": ["#6C1D45", "#99D6EA", "#FADADD"],  # Home: Claret | Away: Sky Blue
    "Chelsea": ["#034694", "#FFFFFF", "#D1D3D4"],  # Home: Blue | Away: Silver/Grey
    "Crystal Palace": [
        "#1B458F",
        "#C4122E",
        "#E5E7EB",
    ],  # Home: Blue/Red | Away: Light Blue
    "Everton": ["#003399", "#FFFFFF", "#FFD100"],  # Home: Blue | Away: Yellow
    "Fulham": ["#F4F4F4", "#111111", "#CC0000"],  # Home: White | Away: Black/Red
    "Leeds United": ["#FFFFFF", "#1D428A", "#FFCD00"],  # Home: White | Away: Blue
    "Liverpool": ["#C8102E", "#22C55E", "#F6EB61"],  # Home: Red | Away: Teal/Yellow
    "Manchester City": ["#6CABDD", "#FFFFFF", "#1C2C5B"],  # Home: Sky Blue | Away: Navy
    "Manchester United": ["#DA291C", "#FBE122", "#000000"],  # Home: Red | Away: Black
    "Newcastle": ["#2D2D2D", "#FFFFFF", "#A3A3A3"],  # Home: Black/White | Away: Blue
    "Newcastle United": [
        "#2D2D2D",
        "#FFFFFF",
        "#A3A3A3",
    ],  # Home: Black/White | Away: Blue
    "Nottm Forest": ["#DD0000", "#FFFFFF", "#FDB913"],  # Home: Red | Away: Yellow
    "Nottingham Forest": ["#DD0000", "#FFFFFF", "#FDB913"],  # Home: Red | Away: Yellow
    "Sunderland": ["#EB172B", "#FFFFFF", "#000000"],  # Home: Red/White | Away: Black
    "Tottenham": ["#FFFFFF", "#132257", "#C0C0C0"],  # Home: White | Away: Navy
    "Tottenham Hotspur": ["#FFFFFF", "#132257", "#C0C0C0"],  # Home: White | Away: Navy
    "West Ham": ["#7A263A", "#F5C542", "#F3D459"],  # Home: Claret/Blue | Away: Yellow
    "West Ham United": [
        "#7A263A",
        "#F5C542",
        "#F3D459",
    ],  # Home: Claret/Blue | Away: Yellow
    "Wolves": ["#FDB913", "#231F20", "#FFFFFF"],  # Home: Gold/Black | Away: White
    "Wolverhampton Wanderers": ["#FDB913", "#231F20", "#FFFFFF"],
    # LaLiga EA Sports 2025/26
    "Athletic Club": ["#EE2523", "#FFFFFF", "#111111"],  # Home: Red/White | Away: Black
    "Athletic Bilbao": [
        "#EE2523",
        "#FFFFFF",
        "#111111",
    ],  # Home: Red/White | Away: Black
    "Atletico Madrid": ["#CB3524", "#FFFFFF", "#262B59"],  # Home: Red/Blue | Away: Navy
    "Atlético de Madrid": [
        "#CB3524",
        "#FFFFFF",
        "#262B59",
    ],  # Home: Red/Blue | Away: Navy
    "Atlético Madrid": ["#CB3524", "#FFFFFF", "#262B59"],  # Home: Red/Blue | Away: Navy
    "CA Osasuna": ["#0A346F", "#D91E2E", "#FFFFFF"],  # Home: Navy/Red | Away: White
    "Osasuna": ["#0A346F", "#D91E2E", "#FFFFFF"],  # Home: Navy/Red | Away: White
    "Celta": ["#8AC3EE", "#FFFFFF", "#C8102E"],  # Home: Light Blue | Away: Red
    "Celta Vigo": ["#8AC3EE", "#FFFFFF", "#C8102E"],  # Home: Light Blue | Away: Red
    "Deportivo Alaves": ["#005BAC", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Deportivo Alavés": ["#005BAC", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Alaves": ["#005BAC", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Alavés": ["#005BAC", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Elche": ["#FFFFFF", "#007A3D", "#111111"],  # Home: White/Green | Away: Black
    "Elche CF": ["#FFFFFF", "#007A3D", "#111111"],  # Home: White/Green | Away: Black
    "Barcelona": ["#A50044", "#004D98", "#EDBB00"],  # Home: Blaugrana | Away: Yellow
    "FC Barcelona": ["#A50044", "#004D98", "#EDBB00"],  # Home: Blaugrana | Away: Yellow
    "Getafe": ["#005999", "#FFFFFF", "#E30613"],  # Home: Blue | Away: Red
    "Getafe CF": ["#005999", "#FFFFFF", "#E30613"],  # Home: Blue | Away: Red
    "Girona": ["#E21D2F", "#FFFFFF", "#111111"],  # Home: Red/White | Away: Black
    "Girona FC": ["#E21D2F", "#FFFFFF", "#111111"],  # Home: Red/White | Away: Black
    "Levante": ["#B0043C", "#005BBB", "#FFFFFF"],  # Home: Granota | Away: Blue
    "Levante UD": ["#B0043C", "#005BBB", "#FFFFFF"],  # Home: Granota | Away: Blue
    "Rayo Vallecano": [
        "#FFFFFF",
        "#D71920",
        "#111111",
    ],  # Home: White/Red | Away: Black
    "Espanyol": ["#0072CE", "#FFFFFF", "#111111"],  # Home: Blue/White | Away: Black
    "RCD Espanyol": ["#0072CE", "#FFFFFF", "#111111"],  # Home: Blue/White | Away: Black
    "Mallorca": ["#E30613", "#111111", "#F7C600"],  # Home: Red/Black | Away: Yellow
    "RCD Mallorca": ["#E30613", "#111111", "#F7C600"],  # Home: Red/Black | Away: Yellow
    "Real Betis": ["#00843D", "#FFFFFF", "#111111"],  # Home: Green | Away: Black
    "Betis": ["#00843D", "#FFFFFF", "#111111"],  # Home: Green | Away: Black
    "Real Madrid": ["#FFFFFF", "#FEBE10", "#00529F"],  # Home: White | Away: Blue
    "Real Oviedo": ["#00529F", "#FFFFFF", "#F7C600"],  # Home: Blue | Away: Yellow
    "Real Sociedad": [
        "#0067B1",
        "#FFFFFF",
        "#111111",
    ],  # Home: Blue/White | Away: Black
    "Sevilla": ["#FFFFFF", "#D71920", "#111111"],  # Home: White/Red | Away: Black
    "Sevilla FC": ["#FFFFFF", "#D71920", "#111111"],  # Home: White/Red | Away: Black
    "Valencia": ["#FFFFFF", "#F58220", "#111111"],  # Home: White/Orange | Away: Black
    "Valencia CF": [
        "#FFFFFF",
        "#F58220",
        "#111111",
    ],  # Home: White/Orange | Away: Black
    "Villarreal": ["#F5DD02", "#005BAC", "#111111"],  # Home: Yellow | Away: Blue
    "Villarreal CF": ["#F5DD02", "#005BAC", "#111111"],  # Home: Yellow | Away: Blue
    # Serie A 2025/26
    "Atalanta": ["#1D3C6A", "#111111", "#FFFFFF"],  # Home: Navy/Black | Away: White
    "Bologna": ["#1B365D", "#DA291C", "#FFFFFF"],  # Home: Navy/Red | Away: White
    "Cagliari": ["#0B2B5C", "#B5121B", "#F6D4A1"],  # Home: Navy/Red | Away: Cream
    "Como": ["#005CA8", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Cremonese": ["#8A1538", "#A7A8AA", "#FFFFFF"],  # Home: Grey/Red | Away: White
    "Fiorentina": ["#5A1A8B", "#FFFFFF", "#D4AF37"],  # Home: Purple | Away: Gold
    "Genoa": ["#0E2240", "#B5121B", "#FFFFFF"],  # Home: Navy/Red | Away: White
    "Hellas Verona": [
        "#002F6C",
        "#F7C600",
        "#FFFFFF",
    ],  # Home: Blue/Yellow | Away: White
    "Inter": ["#010E80", "#0068B5", "#111111"],  # Home: Blue/Black | Away: Black
    "Inter Milan": ["#010E80", "#0068B5", "#111111"],  # Home: Blue/Black | Away: Black
    "Juventus": ["#FFFFFF", "#111111", "#FBCB05"],  # Home: White/Black | Away: Gold
    "Lazio": ["#87CEEB", "#FFFFFF", "#0B2240"],  # Home: Sky Blue | Away: Navy
    "Lecce": ["#D71920", "#F7C600", "#0057B8"],  # Home: Red/Yellow | Away: Blue
    "AC Milan": ["#FB090B", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Milan": ["#FB090B", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Napoli": ["#12A8E0", "#FFFFFF", "#111111"],  # Home: Light Blue | Away: Black
    "Parma": ["#FFFFFF", "#003DA5", "#FECB00"],  # Home: White/Blue | Away: Yellow
    "Pisa": ["#00205B", "#111111", "#D4AF37"],  # Home: Navy | Away: Gold
    "Roma": ["#8E1F2F", "#F9B233", "#111111"],  # Home: Giallorosso | Away: Black
    "Sassuolo": ["#009A44", "#111111", "#FFFFFF"],  # Home: Green/Black | Away: White
    "Torino": ["#7C2D2D", "#FFFFFF", "#D4AF37"],  # Home: Maroon | Away: Gold
    "Udinese": ["#FFFFFF", "#111111", "#A6A6A6"],  # Home: White/Black | Away: Grey
    # Bundesliga 2025/26
    "Bayern Munich": ["#DC052D", "#FFFFFF", "#0066B2"],  # Home: Red | Away: Blue
    "FC Bayern Munich": ["#DC052D", "#FFFFFF", "#0066B2"],  # Home: Red | Away: Blue
    "Borussia Dortmund": [
        "#FDE100",
        "#111111",
        "#FFFFFF",
    ],  # Home: Yellow/Black | Away: White
    "Dortmund": ["#FDE100", "#111111", "#FFFFFF"],  # Home: Yellow/Black | Away: White
    "RB Leipzig": ["#FFFFFF", "#DD0741", "#0C2340"],  # Home: White/Red | Away: Navy
    "Leipzig": ["#FFFFFF", "#DD0741", "#0C2340"],  # Home: White/Red | Away: Navy
    "VfB Stuttgart": ["#FFFFFF", "#E32219", "#111111"],  # Home: White/Red | Away: Black
    "Stuttgart": ["#FFFFFF", "#E32219", "#111111"],  # Home: White/Red | Away: Black
    "Hoffenheim": ["#0057B8", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "TSG Hoffenheim": ["#0057B8", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Bayer Leverkusen": [
        "#E32221",
        "#111111",
        "#FFFFFF",
    ],  # Home: Red/Black | Away: White
    "Leverkusen": ["#E32221", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Eintracht Frankfurt": [
        "#E1000F",
        "#111111",
        "#FFFFFF",
    ],  # Home: Red/Black | Away: White
    "Frankfurt": ["#E1000F", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Freiburg": ["#D50032", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "SC Freiburg": ["#D50032", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Augsburg": ["#BA0C2F", "#007A33", "#FFFFFF"],  # Home: Red/Green | Away: White
    "Mainz": ["#C31432", "#FFFFFF", "#111111"],  # Home: Red | Away: Black
    "Mainz 05": ["#C31432", "#FFFFFF", "#111111"],  # Home: Red | Away: Black
    "Borussia Mönchengladbach": [
        "#FFFFFF",
        "#00843D",
        "#111111",
    ],  # Home: White/Green | Away: Black
    "M'gladbach": ["#FFFFFF", "#00843D", "#111111"],  # Home: White/Green | Away: Black
    "Borussia Monchengladbach": [
        "#FFFFFF",
        "#00843D",
        "#111111",
    ],  # Home: White/Green | Away: Black
    "Werder Bremen": ["#00843D", "#FFFFFF", "#F7C600"],  # Home: Green | Away: Yellow
    "Union Berlin": ["#D00000", "#F7C600", "#FFFFFF"],  # Home: Red/Yellow | Away: White
    "Cologne": ["#FFFFFF", "#ED1C24", "#111111"],  # Home: White/Red | Away: Black
    "FC Koln": ["#FFFFFF", "#ED1C24", "#111111"],  # Home: White/Red | Away: Black
    "FC Köln": ["#FFFFFF", "#ED1C24", "#111111"],  # Home: White/Red | Away: Black
    "Hamburg": ["#005CA9", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Hamburger SV": ["#005CA9", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "St. Pauli": ["#5B3A29", "#FFFFFF", "#D71920"],  # Home: Brown | Away: Red
    "Wolfsburg": ["#65B32E", "#FFFFFF", "#111111"],  # Home: Green | Away: Black
    "Heidenheim": ["#E30613", "#005BAC", "#FFFFFF"],  # Home: Red/Blue | Away: White
    # Ligue 1 2025/26
    "Angers": ["#FFFFFF", "#111111", "#D4AF37"],  # Home: White/Black | Away: Gold
    "SCO Angers": ["#FFFFFF", "#111111", "#D4AF37"],  # Home: White/Black | Away: Gold
    "Auxerre": ["#0057B8", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "AJ Auxerre": ["#0057B8", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Brest": ["#E30613", "#FFFFFF", "#111111"],  # Home: Red | Away: Black
    "Stade Brestois": ["#E30613", "#FFFFFF", "#111111"],  # Home: Red | Away: Black
    "Le Havre": ["#6CB4EE", "#0B2B5C", "#FFFFFF"],  # Home: Sky Blue/Navy | Away: White
    "Havre AC": ["#6CB4EE", "#0B2B5C", "#FFFFFF"],  # Home: Sky Blue/Navy | Away: White
    "Lens": ["#FFD100", "#E30613", "#111111"],  # Home: Yellow/Red | Away: Black
    "RC Lens": ["#FFD100", "#E30613", "#111111"],  # Home: Yellow/Red | Away: Black
    "Lille": ["#E01E37", "#0B1F3A", "#FFFFFF"],  # Home: Red/Navy | Away: White
    "LOSC": ["#E01E37", "#0B1F3A", "#FFFFFF"],  # Home: Red/Navy | Away: White
    "Lorient": ["#F58220", "#111111", "#FFFFFF"],  # Home: Orange/Black | Away: White
    "Metz": ["#8A1538", "#FFFFFF", "#111111"],  # Home: Maroon | Away: Black
    "FC Metz": ["#8A1538", "#FFFFFF", "#111111"],  # Home: Maroon | Away: Black
    "Lyon": ["#FFFFFF", "#003DA5", "#D71920"],  # Home: White/Blue | Away: Red
    "Olympique Lyonnais": [
        "#FFFFFF",
        "#003DA5",
        "#D71920",
    ],  # Home: White/Blue | Away: Red
    "Marseille": ["#FFFFFF", "#00A3E0", "#111111"],  # Home: White/Blue | Away: Black
    "Olympique de Marseille": [
        "#FFFFFF",
        "#00A3E0",
        "#111111",
    ],  # Home: White/Blue | Away: Black
    "Monaco": ["#FFFFFF", "#E30613", "#C0C0C0"],  # Home: White/Red | Away: Silver
    "AS Monaco": ["#FFFFFF", "#E30613", "#C0C0C0"],  # Home: White/Red | Away: Silver
    "Nantes": ["#FFE500", "#00843D", "#111111"],  # Home: Yellow/Green | Away: Black
    "FC Nantes": ["#FFE500", "#00843D", "#111111"],  # Home: Yellow/Green | Away: Black
    "Nice": ["#D71920", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "OGC Nice": ["#D71920", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Paris FC": ["#132257", "#8AC3EE", "#FFFFFF"],  # Home: Navy | Away: White
    "PSG": ["#004170", "#DA291C", "#FFFFFF"],  # Home: Navy/Red | Away: White
    "Paris Saint-Germain": [
        "#004170",
        "#DA291C",
        "#FFFFFF",
    ],  # Home: Navy/Red | Away: White
    "Rennes": ["#E30613", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Stade Rennais": ["#E30613", "#111111", "#FFFFFF"],  # Home: Red/Black | Away: White
    "Strasbourg": ["#00A3E0", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "RC Strasbourg": ["#00A3E0", "#FFFFFF", "#111111"],  # Home: Blue | Away: Black
    "Toulouse": ["#5B2C83", "#FFFFFF", "#D71920"],  # Home: Purple | Away: Red
    "Toulouse FC": ["#5B2C83", "#FFFFFF", "#D71920"],  # Home: Purple | Away: Red
}

INTERNATIONAL_TEAM_PALETTES = {
    "Mexico": ["#006847", "#CE1126", "#FFFFFF"],
    "South Africa": ["#FFB81C", "#007A4D", "#003B5C"],
    "Egypt": ["#CE1126", "#FFFFFF", "#000000"],
    "Argentina": ["#75AADB", "#FFFFFF", "#F6B40E"],
    "Brazil": ["#FFDF00", "#009C3B", "#002776"],
    "France": ["#0055A4", "#FFFFFF", "#EF4135"],
    "England": ["#FFFFFF", "#C8102E", "#00247D"],
    "Spain": ["#AA151B", "#F1BF00", "#002B7F"],
    "Germany": ["#FFFFFF", "#000000", "#DD0000"],
    "Italy": ["#0066B3", "#FFFFFF", "#008C45"],
    "Portugal": ["#C8102E", "#006600", "#FFCC00"],
    "Netherlands": ["#F36C21", "#21468B", "#FFFFFF"],
    "Belgium": ["#ED2939", "#FAE042", "#000000"],
    "Morocco": ["#C1272D", "#006233", "#FFFFFF"],
    "United States": ["#3C3B6E", "#B22234", "#FFFFFF"],
    "USA": ["#3C3B6E", "#B22234", "#FFFFFF"],
    "Canada": ["#FF0000", "#FFFFFF", "#111111"],
    "Japan": ["#1B365D", "#BC002D", "#FFFFFF"],
    "Saudi Arabia": ["#006C35", "#FFFFFF", "#111111"],
    "Qatar": ["#8A1538", "#FFFFFF", "#111111"],
    "Nigeria": ["#008751", "#FFFFFF", "#111111"],
    "Ghana": ["#FFFFFF", "#CE1126", "#006B3F"],
    "Senegal": ["#00853F", "#FDEF42", "#E31B23"],
    "Uruguay": ["#6CABDD", "#FFFFFF", "#FCD116"],
    "Colombia": ["#FCD116", "#003893", "#CE1126"],
    "Croatia": ["#FF0000", "#FFFFFF", "#171796"],
    "Switzerland": ["#D52B1E", "#FFFFFF", "#111111"],
    "Denmark": ["#C60C30", "#FFFFFF", "#111111"],
    "Poland": ["#FFFFFF", "#DC143C", "#111111"],
    "Australia": ["#FFCD00", "#00843D", "#012169"],
    "Korea Republic": ["#C60C30", "#003478", "#FFFFFF"],
    "South Korea": ["#C60C30", "#003478", "#FFFFFF"],
    "Ivory Coast": ["#F77F00", "#009E60", "#FFFFFF"],
    "Côte d'Ivoire": ["#F77F00", "#009E60", "#FFFFFF"],
    "Cameroon": ["#007A5E", "#CE1126", "#FCD116"],
    "Tunisia": ["#E70013", "#FFFFFF", "#000000"],
    "Sweden": ["#FECB00", "#006AA7", "#1D3557"],
    "Algeria": ["#006233", "#FFFFFF", "#D21034"],
    "Turkey": ["#E30A17", "#FFFFFF", "#111111"],
    "Chile": ["#D52B1E", "#0039A6", "#FFFFFF"],
    "Peru": ["#D91023", "#FFFFFF", "#111111"],
    "Serbia": ["#C6363C", "#0C4076", "#FFFFFF"],
    "Scotland": ["#005EB8", "#FFFFFF", "#111111"],
    "Wales": ["#D30731", "#00AD36", "#FFFFFF"],
    "Norway": ["#BA0C2F", "#00205B", "#FFFFFF"],
    "DR Congo": ["#007FFF", "#F7D618", "#CE1021"],
    "Congo DR": ["#007FFF", "#F7D618", "#CE1021"],
    "Democratic Republic of the Congo": ["#007FFF", "#F7D618", "#CE1021"],
    "Congo": ["#009543", "#FBDE4A", "#DC241F"],
    "Angola": ["#CC092F", "#F7D618", "#111111"],
    "Cape Verde": ["#003893", "#FFFFFF", "#CF2027"],
    "Mali": ["#14B53A", "#FCD116", "#CE1126"],
    "Burkina Faso": ["#EF2B2D", "#009E49", "#FCD116"],
    "Jamaica": ["#009B3A", "#FED100", "#111111"],
    "Paraguay": ["#DA121A", "#FFFFFF", "#0038A8"],
    "Bolivia": ["#007A33", "#FFD100", "#D52B1E"],
    "Ecuador": ["#FFDD00", "#0072CE", "#ED1C24"],
    "Venezuela": ["#7A1E3C", "#FFFFFF", "#A50021"],
    "Costa Rica": ["#CE1126", "#FFFFFF", "#002B7F"],
    "Panama": ["#D21034", "#0072CE", "#FFFFFF"],
    "Honduras": ["#0073CF", "#FFFFFF", "#111111"],
    "Jordan": ["#CE1126", "#000000", "#FFFFFF"],
    "Iraq": ["#CE1126", "#FFFFFF", "#000000"],
    "Iran": ["#239F40", "#FFFFFF", "#DA0000"],
    "UAE": ["#FF0000", "#FFFFFF", "#000000"],
    "United Arab Emirates": ["#FF0000", "#FFFFFF", "#000000"],
    "Oman": ["#C8102E", "#FFFFFF", "#007A3D"],
    "Bahrain": ["#CE1126", "#FFFFFF", "#111111"],
    "Kuwait": ["#007A3D", "#CE1126", "#FFFFFF"],
    "Syria": ["#CE1126", "#FFFFFF", "#000000"],
    "Uzbekistan": ["#1EB53A", "#0099B5", "#FFFFFF"],
    "China": ["#DE2910", "#FFDE00", "#FFFFFF"],
    "China PR": ["#DE2910", "#FFDE00", "#FFFFFF"],
    "India": ["#FF671F", "#06038D", "#FFFFFF"],
    "New Zealand": ["#FFFFFF", "#111111", "#CE1126"],
    "Iceland": ["#02529C", "#FFFFFF", "#DC1E35"],
    "Finland": ["#003580", "#FFFFFF", "#111111"],
    "Austria": ["#ED2939", "#FFFFFF", "#111111"],
    "Czech Republic": ["#D7141A", "#11457E", "#FFFFFF"],
    "Czechia": ["#D7141A", "#11457E", "#FFFFFF"],
    "Slovakia": ["#0B4EA2", "#EE1C25", "#FFFFFF"],
    "Slovenia": ["#FFFFFF", "#005DA4", "#ED1C24"],
    "Romania": ["#FCD116", "#002B7F", "#CE1126"],
    "Hungary": ["#CE1126", "#FFFFFF", "#007A3D"],
    "Ukraine": ["#FFD500", "#005BBB", "#111111"],
    "Greece": ["#0D5EAF", "#FFFFFF", "#111111"],
    "Israel": ["#0038B8", "#FFFFFF", "#111111"],
    "Albania": ["#E41E20", "#000000", "#FFFFFF"],
    "Georgia": ["#FFFFFF", "#E4002B", "#111111"],
    "Kosovo": ["#3B5BA5", "#FFFFFF", "#FFD700"],
    "Northern Ireland": ["#FFFFFF", "#005EB8", "#111111"],
    "Republic of Ireland": ["#169B62", "#FFFFFF", "#FF883E"],
    "Ireland": ["#169B62", "#FFFFFF", "#FF883E"],
    "Finland women": ["#003580", "#FFFFFF", "#111111"],
    "Gabon": ["#009E60", "#FCD116", "#3A75C4"],
    "Zambia": ["#198A00", "#EF3340", "#000000"],
    "Guinea": ["#CE1126", "#FCD116", "#009460"],
    "Equatorial Guinea": ["#3E9A00", "#E32118", "#0073CF"],
    "Benin": ["#008751", "#FCD116", "#E8112D"],
    "Namibia": ["#003580", "#D21034", "#009543"],
    "Mauritania": ["#00A95C", "#FFC400", "#D01C1F"],
}

TOP5_2025_26_TEAM_PALETTES.update(INTERNATIONAL_TEAM_PALETTES)

# ── Extra clubs & national teams (real home-kit colours) ─────────────────
# Beyond the big-5 leagues: Primeira Liga, Eredivisie, Scottish PL, Süper Lig,
# Saudi Pro League, Belgian PL, Brazil, Argentina, MLS, Egypt, plus other UEFA
# regulars — and additional national teams. First colour = home shirt.
EXTRA_TEAM_PALETTES = {
    # Primeira Liga (Portugal)
    "Benfica": ["#E50914", "#FFFFFF", "#000000"],
    "SL Benfica": ["#E50914", "#FFFFFF", "#000000"],
    "Porto": ["#003F87", "#FFFFFF", "#000000"],
    "FC Porto": ["#003F87", "#FFFFFF", "#000000"],
    "Sporting CP": ["#008057", "#FFFFFF", "#000000"],
    "Sporting": ["#008057", "#FFFFFF", "#000000"],
    "Sporting Lisbon": ["#008057", "#FFFFFF", "#000000"],
    "Braga": ["#C8102E", "#FFFFFF", "#000000"],
    "SC Braga": ["#C8102E", "#FFFFFF", "#000000"],
    "Vitoria Guimaraes": ["#FFFFFF", "#000000", "#D4AF37"],
    # Eredivisie (Netherlands)
    "Ajax": ["#D2122E", "#FFFFFF", "#000000"],
    "PSV": ["#ED1C24", "#FFFFFF", "#000000"],
    "PSV Eindhoven": ["#ED1C24", "#FFFFFF", "#000000"],
    "Feyenoord": ["#C8102E", "#FFFFFF", "#000000"],
    "AZ": ["#E2001A", "#FFFFFF", "#000000"],
    "AZ Alkmaar": ["#E2001A", "#FFFFFF", "#000000"],
    "Twente": ["#E2001A", "#FFFFFF", "#000000"],
    # Scottish Premiership
    "Celtic": ["#018749", "#FFFFFF", "#000000"],
    "Rangers": ["#1B458F", "#FFFFFF", "#E4002B"],
    "Aberdeen": ["#E2001A", "#FFFFFF", "#000000"],
    "Heart of Midlothian": ["#7A1F3D", "#F4D03F", "#FFFFFF"],
    "Hibernian": ["#00843D", "#FFFFFF", "#000000"],
    # Süper Lig (Turkey)
    "Galatasaray": ["#A90432", "#FDB912", "#000000"],
    "Fenerbahce": ["#15326B", "#FFED00", "#FFFFFF"],
    "Fenerbahçe": ["#15326B", "#FFED00", "#FFFFFF"],
    "Besiktas": ["#000000", "#FFFFFF", "#E30613"],
    "Beşiktaş": ["#000000", "#FFFFFF", "#E30613"],
    "Trabzonspor": ["#7A1632", "#1E9BD7", "#FFFFFF"],
    # Saudi Pro League
    "Al Hilal": ["#0A4DA2", "#FFFFFF", "#000000"],
    "Al Nassr": ["#FEDD00", "#1A3E8B", "#FFFFFF"],
    "Al Ittihad": ["#F4C400", "#000000", "#FFFFFF"],
    "Al Ahli Saudi": ["#007A33", "#FFFFFF", "#000000"],
    "Al Shabab": ["#FFFFFF", "#000000", "#D4AF37"],
    "Al Ettifaq": ["#C8102E", "#FFFFFF", "#000000"],
    # Belgian Pro League
    "Club Brugge": ["#0E63B0", "#000000", "#FFFFFF"],
    "Anderlecht": ["#52247F", "#FFFFFF", "#000000"],
    "Genk": ["#0050A0", "#FFFFFF", "#000000"],
    "Gent": ["#005CA9", "#FFFFFF", "#000000"],
    "Standard Liege": ["#E2001A", "#FFFFFF", "#000000"],
    "Royal Antwerp": ["#D2122E", "#FFFFFF", "#000000"],
    "Union SG": ["#FFD200", "#0A2342", "#FFFFFF"],
    # Brazil (Série A)
    "Flamengo": ["#C52613", "#000000", "#FFFFFF"],
    "Palmeiras": ["#006437", "#FFFFFF", "#000000"],
    "Corinthians": ["#FFFFFF", "#000000", "#CCCCCC"],
    "Sao Paulo": ["#FFFFFF", "#E30613", "#000000"],
    "São Paulo": ["#FFFFFF", "#E30613", "#000000"],
    "Santos": ["#FFFFFF", "#000000", "#CCCCCC"],
    "Fluminense": ["#7A1631", "#1B5E20", "#FFFFFF"],
    "Botafogo": ["#000000", "#FFFFFF", "#C0C0C0"],
    "Gremio": ["#0D80BF", "#000000", "#FFFFFF"],
    "Grêmio": ["#0D80BF", "#000000", "#FFFFFF"],
    "Internacional": ["#E5050F", "#FFFFFF", "#000000"],
    "Vasco da Gama": ["#000000", "#FFFFFF", "#E30613"],
    "Atletico Mineiro": ["#000000", "#FFFFFF", "#C0C0C0"],
    "Atlético Mineiro": ["#000000", "#FFFFFF", "#C0C0C0"],
    "Cruzeiro": ["#0033A0", "#FFFFFF", "#000000"],
    # Argentina (Primera División)
    "Boca Juniors": ["#0A3E7E", "#F2C300", "#FFFFFF"],
    "River Plate": ["#FFFFFF", "#E1122B", "#000000"],
    "Racing Club": ["#6CACE4", "#FFFFFF", "#000000"],
    "Independiente": ["#E2001A", "#FFFFFF", "#000000"],
    "San Lorenzo": ["#0A2342", "#E2001A", "#FFFFFF"],
    "Velez Sarsfield": ["#FFFFFF", "#003DA5", "#000000"],
    # MLS
    "Inter Miami": ["#F4B5CB", "#000000", "#231F20"],
    "Inter Miami CF": ["#F4B5CB", "#000000", "#231F20"],
    "LAFC": ["#000000", "#C39E6D", "#FFFFFF"],
    "Los Angeles FC": ["#000000", "#C39E6D", "#FFFFFF"],
    "LA Galaxy": ["#00245D", "#FFFFFF", "#FCD303"],
    "Seattle Sounders": ["#5D9741", "#1E1E5C", "#FFFFFF"],
    "Atlanta United": ["#A6192E", "#000000", "#C39E6D"],
    # Egypt
    "Al Ahly": ["#E30613", "#FFFFFF", "#000000"],
    "Al Ahly SC": ["#E30613", "#FFFFFF", "#000000"],
    "Zamalek": ["#FFFFFF", "#D2122E", "#000000"],
    "Zamalek SC": ["#FFFFFF", "#D2122E", "#000000"],
    "Pyramids FC": ["#1B3A8C", "#FFFFFF", "#000000"],
    # Other UEFA regulars
    "Shakhtar Donetsk": ["#F58220", "#000000", "#FFFFFF"],
    "Dynamo Kyiv": ["#FFFFFF", "#005BBB", "#000000"],
    "Red Bull Salzburg": ["#D3041E", "#FFFFFF", "#000000"],
    "RB Salzburg": ["#D3041E", "#FFFFFF", "#000000"],
    "Salzburg": ["#D3041E", "#FFFFFF", "#000000"],
    "Olympiacos": ["#E30613", "#FFFFFF", "#000000"],
    "Panathinaikos": ["#007A33", "#FFFFFF", "#000000"],
    "PAOK": ["#000000", "#FFFFFF", "#C0C0C0"],
    "Slavia Prague": ["#D7141A", "#FFFFFF", "#000000"],
    "Sparta Prague": ["#9B1B30", "#FFFFFF", "#000000"],
    "Young Boys": ["#FFE500", "#000000", "#FFFFFF"],
    "FC Copenhagen": ["#FFFFFF", "#163C6E", "#000000"],
    "Sturm Graz": ["#000000", "#FFFFFF", "#C0C0C0"],
    "Bodo/Glimt": ["#FFE500", "#000000", "#FFFFFF"],
    # Additional national teams
    "North Macedonia": ["#D20000", "#FFE600", "#FFFFFF"],
    "Montenegro": ["#C40308", "#D4AF37", "#FFFFFF"],
    "Bosnia and Herzegovina": ["#002395", "#FCD116", "#FFFFFF"],
    "Bosnia": ["#002395", "#FCD116", "#FFFFFF"],
    "Bulgaria": ["#FFFFFF", "#00966E", "#D62612"],
    "Armenia": ["#D90012", "#0033A0", "#F2A800"],
    "Azerbaijan": ["#00B5E2", "#EF3340", "#509E2F"],
    "Luxembourg": ["#ED2939", "#FFFFFF", "#00A1DE"],
    "Cyprus": ["#005CBF", "#FFFFFF", "#D57800"],
    "Estonia": ["#0072CE", "#000000", "#FFFFFF"],
    "Latvia": ["#9E3039", "#FFFFFF", "#000000"],
    "Lithuania": ["#FDB913", "#006A44", "#C1272D"],
    "Belarus": ["#D22730", "#00843D", "#FFFFFF"],
    "Moldova": ["#0046AE", "#FFD200", "#CC092F"],
    "Malta": ["#CF142B", "#FFFFFF", "#000000"],
    "Faroe Islands": ["#FFFFFF", "#0065BD", "#ED2939"],
    "Lebanon": ["#ED1C24", "#FFFFFF", "#00A651"],
    "Palestine": ["#000000", "#FFFFFF", "#CE1126"],
    "Libya": ["#239E46", "#000000", "#E70013"],
    "Sudan": ["#D21034", "#FFFFFF", "#007229"],
    "Vietnam": ["#DA251D", "#FFFF00", "#FFFFFF"],
    "Thailand": ["#241D4F", "#A51931", "#FFFFFF"],
    "Indonesia": ["#CE1126", "#FFFFFF", "#000000"],
    "Malaysia": ["#FCD116", "#000000", "#CC0001"],
    "El Salvador": ["#0F47AF", "#FFFFFF", "#000000"],
    "Guatemala": ["#4997D0", "#FFFFFF", "#000000"],
    "Haiti": ["#00209F", "#D21034", "#FFFFFF"],
    "Trinidad and Tobago": ["#E00000", "#000000", "#FFFFFF"],
    "Curacao": ["#002B7F", "#F9D616", "#FFFFFF"],
    "Curaçao": ["#002B7F", "#F9D616", "#FFFFFF"],
    "Suriname": ["#377E3F", "#FFFFFF", "#B40A2D"],
    "New Caledonia": ["#00A1DE", "#FFFFFF", "#ED2939"],
    "Tahiti": ["#FFFFFF", "#E30613", "#000000"],
}
TOP5_2025_26_TEAM_PALETTES.update(EXTRA_TEAM_PALETTES)

# Make the primary colour table cover all new teams while preserving earlier explicit values.
for _club_name, _palette in TOP5_2025_26_TEAM_PALETTES.items():
    if _palette:
        TEAM_COLORS[_club_name] = _palette[0]

# Aliases used by WhoScored / Opta / common spellings.
TEAM_ALIASES.update(
    {
        "mexico": "Mexico",
        "méxico": "Mexico",
        "south africa": "South Africa",
        "bafana bafana": "South Africa",
        "egypt": "Egypt",
        "morocco": "Morocco",
        "sweden": "Sweden",
        "tunisia": "Tunisia",
        "united states of america": "United States",
        "united states": "United States",
        "usa": "USA",
        "usmnt": "USA",
        "korea republic": "Korea Republic",
        "south korea": "South Korea",
        "cote d'ivoire": "Ivory Coast",
        "côte d'ivoire": "Côte d'Ivoire",
        "ivory coast": "Ivory Coast",
        "dr congo": "DR Congo",
        "d r congo": "DR Congo",
        "congo dr": "Congo DR",
        "democratic republic of congo": "Democratic Republic of the Congo",
        "democratic republic of the congo": "Democratic Republic of the Congo",
        "newcastle united": "Newcastle",
        "nottingham forest": "Nottm Forest",
        "nottm forest": "Nottm Forest",
        "leeds": "Leeds United",
        "leeds united": "Leeds United",
        "sunderland": "Sunderland",
        "burnley": "Burnley",
        "wolves": "Wolves",
        "wolverhampton wanderers": "Wolves",
        "athletic bilbao": "Athletic Club",
        "athletic club": "Athletic Club",
        "atletico madrid": "Atletico Madrid",
        "atlético de madrid": "Atletico Madrid",
        "osasuna": "CA Osasuna",
        "celta vigo": "Celta",
        "alaves": "Deportivo Alaves",
        "alavés": "Deportivo Alaves",
        "deportivo alavés": "Deportivo Alaves",
        "elche cf": "Elche",
        "girona fc": "Girona",
        "levante ud": "Levante",
        "rcd espanyol": "Espanyol",
        "rcd mallorca": "Mallorca",
        "real betis": "Real Betis",
        "betis": "Real Betis",
        "real oviedo": "Real Oviedo",
        "real sociedad": "Real Sociedad",
        "sevilla fc": "Sevilla",
        "valencia cf": "Valencia",
        "villarreal cf": "Villarreal",
        "inter": "Inter Milan",
        "fc bayern munich": "Bayern Munich",
        "leipzig": "RB Leipzig",
        "bayer 04 leverkusen": "Bayer Leverkusen",
        "leverkusen": "Bayer Leverkusen",
        "m'gladbach": "Borussia Mönchengladbach",
        "borussia monchengladbach": "Borussia Mönchengladbach",
        "koln": "Cologne",
        "köln": "Cologne",
        "fc koln": "Cologne",
        "fc köln": "Cologne",
        "hamburger sv": "Hamburg",
        "sco angers": "Angers",
        "aj auxerre": "Auxerre",
        "stade brestois": "Brest",
        "havre ac": "Le Havre",
        "rc lens": "Lens",
        "losc": "Lille",
        "fc metz": "Metz",
        "olympique lyonnais": "Lyon",
        "olympique de marseille": "Marseille",
        "om": "Marseille",
        "as monaco": "Monaco",
        "fc nantes": "Nantes",
        "ogc nice": "Nice",
        "stade rennais": "Rennes",
        "rc strasbourg": "Strasbourg",
        "toulouse fc": "Toulouse",
    }
)


DEFAULT_HOME = "#4D8DFF"
DEFAULT_AWAY = "#FF4D4D"


def get_team_color(team_name: str, fallback: str) -> str:
    """
    Return a team color from TEAM_COLORS.
    Search in the following order:
      1. Exact case-insensitive match
      2. Alias lookup in TEAM_ALIASES
      3. Partial match in either direction
    """
    if not team_name:
        return fallback
    name_lc = team_name.strip().lower()

    for key, color in TEAM_COLORS.items():
        if key.lower() == name_lc:
            return color

    # 2) alias
    if name_lc in TEAM_ALIASES:
        alias_target = TEAM_ALIASES[name_lc]
        return TEAM_COLORS.get(alias_target, fallback)

    for key, color in TEAM_COLORS.items():
        key_lc = key.lower()
        if key_lc in name_lc or name_lc in key_lc:
            return color

    return fallback


# ── Colour contrast helpers ─────────────────────────────────────────
def _hex_to_rgb01(color: str):
    """Return RGB in 0..1 for a hex colour; defaults to black on bad input."""
    try:
        c = str(color or "").strip()
        if c.startswith("#"):
            c = c[1:]
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        return int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0, int(c[4:6], 16) / 255.0
    except Exception:
        return 0.0, 0.0, 0.0


def _relative_luminance(color: str) -> float:
    def lin(v):
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = _hex_to_rgb01(color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _is_light_color(color: str) -> bool:
    """Return True for colours light enough to need dark text on top of them."""
    lum = _relative_luminance(color)
    if lum >= 0.45:
        return True
    # Also catch warm/yellow hues (high R+G, low B) that look light visually
    r, g, b = _hex_to_rgb01(color)
    if lum >= 0.35 and r >= 0.70 and g >= 0.55 and b <= 0.35:
        return True
    return False


def _text_on_color(color: str, light: str = "#ffffff", dark: str = "#111827") -> str:
    """Pick readable text for labels placed ON a coloured background.

    Dynamic WCAG-aware contrast logic:
    - Very light colours (luminance ≥ 0.65) always get dark text.
    - Warm/yellow hues (high R+G, low B, lum ≥ 0.35) always get dark text —
      this catches Wolves gold, BVB yellow, Brighton yellow, etc.
    - Medium-light unsaturated colours (lum ≥ 0.50, low saturation) get dark text.
    - Bright cyan/lime at medium luminance get dark text.
    - Everything else (dark, vivid, deep colours) gets white text.
    """
    try:
        lum = _relative_luminance(color)
        r, g, b = _hex_to_rgb01(color)
        maxc, minc = max(r, g, b), min(r, g, b)
        saturation = maxc - minc  # simple chroma in 0..1

        # Rule 1 — pure light / white backgrounds → dark text
        if lum >= 0.65:
            return dark

        # Rule 2 — warm yellows / golds / limes (Wolves, BVB, Brighton away…)
        # High red+green, low blue, noticeable luminance → visually light
        if lum >= 0.35 and r >= 0.70 and g >= 0.55 and b <= 0.35:
            return dark

        # Rule 3 — light greys / off-whites (low saturation, medium-high lum)
        if lum >= 0.50 and saturation < 0.30:
            return dark

        # Rule 4 — saturated greens / cyans at medium luminance
        if lum >= 0.45 and saturation > 0.40 and g >= 0.65:
            return dark

        # Rule 5 — mid-range saturated colours: keep white text
        if lum < 0.56:
            return light

        # Rule 6 — saturated medium lum (e.g. some bright oranges, teals)
        if saturation > 0.30 and lum < 0.68:
            return light

        return dark
    except Exception:
        return light


def _stroke_on_color(color: str) -> str:
    """Outline colour opposite to the chosen text colour."""
    try:
        return "#000000" if _text_on_color(color) == "#ffffff" else "#ffffff"
    except Exception:
        return "#000000"


def _accent_on_color(color: str) -> str:
    """Readable accent for labels placed on coloured team bands."""
    try:
        lum = _relative_luminance(color)
        r, g, b = _hex_to_rgb01(color)
        # Warm/yellow hues or light backgrounds → dark accent
        is_warm_light = lum >= 0.35 and r >= 0.70 and g >= 0.55 and b <= 0.35
        if lum >= 0.55 or is_warm_light:
            return "#111827"
        return "#FFD700"
    except Exception:
        return "#FFD700"


def _canonical_team_name(team_name: str) -> str:
    """Normalize a WhoScored/Opta team name to the colour-table key when possible."""
    if not team_name:
        return ""
    raw = str(team_name).strip()
    raw_lc = raw.lower()
    if raw_lc in TEAM_ALIASES:
        return TEAM_ALIASES[raw_lc]
    for key in TOP5_2025_26_TEAM_PALETTES.keys():
        if key.lower() == raw_lc:
            return key
    for key in TEAM_COLORS.keys():
        if key.lower() == raw_lc:
            return key
    return raw


def _team_palette(team_name: str, fallback: str) -> list[str]:
    """Return the kit palette for a team; always at least one colour."""
    canonical = _canonical_team_name(team_name)
    pal = TOP5_2025_26_TEAM_PALETTES.get(canonical)
    if not pal:
        # Try loose matching for shortened provider names.
        low = canonical.lower()
        for key, vals in TOP5_2025_26_TEAM_PALETTES.items():
            k = key.lower()
            if low and (low in k or k in low):
                pal = vals
                break
    if not pal:
        looked_up = get_team_color(canonical or team_name, fallback)
        if looked_up == fallback:
            # Team is genuinely unrecognized (not in TEAM_COLORS,
            # TOP5_2025_26_TEAM_PALETTES, or INTERNATIONAL_TEAM_PALETTES).
            # Rather than collapsing every unknown side onto the same fixed
            # DEFAULT_HOME/DEFAULT_AWAY constant — which makes any two
            # unmapped teams (e.g. lower-profile national sides) render
            # with identical or near-identical colours — derive a stable,
            # name-specific colour so each unknown team still gets its own
            # consistent, visually distinct identity across the report.
            pal = [_deterministic_unknown_team_color(canonical or team_name)]
        else:
            pal = [looked_up]
    # Remove duplicates while preserving order.
    out = []
    for c in pal:
        if c and c not in out:
            out.append(c)
    return out or [fallback]


# Visually well-separated colours (in hue) used as a deterministic fallback
# pool for teams that have no entry anywhere in TEAM_COLORS,
# TOP5_2025_26_TEAM_PALETTES, or INTERNATIONAL_TEAM_PALETTES. Picking from a
# spread pool (instead of one fixed constant) means two unrecognized teams
# in the same fixture are very unlikely to land on the same or a visually
# similar colour.
_UNKNOWN_TEAM_COLOR_POOL = [
    "#4D8DFF",  # blue
    "#FF4D4D",  # red
    "#3DDC84",  # green
    "#FFC23C",  # gold
    "#E879F9",  # magenta
    "#38BDF8",  # cyan
    "#F97316",  # orange
    "#A78BFA",  # violet
    "#FACC15",  # yellow
    "#22D3EE",  # turquoise
    "#FB7185",  # rose
    "#84CC16",  # lime
]


def _deterministic_unknown_team_color(team_name: str) -> str:
    """Pick a stable colour for an unrecognized team name from a spread pool.

    Uses a hash of the (normalized) team name so the same team always gets
    the same colour across runs/reports, without every unmapped team
    collapsing onto one shared default.
    """
    key = (team_name or "").strip().lower()
    idx = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % len(
        _UNKNOWN_TEAM_COLOR_POOL
    )
    return _UNKNOWN_TEAM_COLOR_POOL[idx]


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = str(hex_color or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        h = "777777"
    try:
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return (0.45, 0.45, 0.45)


def _color_distance(c1: str, c2: str) -> float:
    """Simple RGB distance. Good enough for avoiding chart colour clashes."""
    r1, g1, b1 = _hex_to_rgb01(c1)
    r2, g2, b2 = _hex_to_rgb01(c2)
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def _relative_luminance(hex_color: str) -> float:
    """WCAG-style relative luminance used for readable labels on coloured bands."""
    r, g, b = _hex_to_rgb01(hex_color)

    def lin(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_ratio_hex(fg: str, bg: str = "#000000") -> float:
    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _readable_kit_candidate(
    team_name: str,
    palette: list[str],
    fallback: str = "#9CA3AF",
    *,
    allow_light: bool = False,
) -> str:
    """Choose a shirt/trim colour that is visible on the navy report background."""
    best = fallback
    best_score = -999.0
    for raw in palette or []:
        c = _usable_on_dark(raw, fallback)
        lum = _relative_luminance(c)
        contrast = _contrast_ratio_hex(c, BG_DARK)
        if lum >= 0.82 and not allow_light:
            score = -10.0
        else:
            # Prefer true kit identity, but require enough contrast on navy.
            score = contrast - (0.35 if lum >= 0.76 else 0.0)
        if score > best_score:
            best, best_score = c, score
    if best_score < 2.05:
        return fallback
    return best


def _usable_on_dark(hex_color: str, fallback: str = "#9CA3AF") -> str:
    """
    Avoid invisible black/near-black on the dark visual background.
    v2: Navy/blue tones with low luminance but bright channels are kept,
    because they are clearly distinguishable from a near-black background.
    Only truly dark colours (near-black in ALL channels) are replaced.
    """
    r, g, b = _hex_to_rgb01(hex_color)
    # A colour is invisible on dark bg only if luminance is very low AND
    # no channel is bright enough to create visual contrast.
    # Navy #132257 has low luminance (0.020) but blue=0.34 → visible.
    # Pure black #000000 has luminance 0 and max channel 0 → invisible.
    if _relative_luminance(hex_color) < 0.035 and max(r, g, b) < 0.18:
        return fallback
    return hex_color


def _visible_on_dark(team_name: str, hex_color: str, fallback: str = "#9CA3AF") -> str:
    """
    Return a colour that is clearly visible on a dark dashboard background.
    v3 fix:
      - Truly dark colours (luminance < 0.04 AND no bright channel) → replaced
        with team accent first, then fallback.
      - Navy/blue tones with low luminance but bright channels are kept
        (e.g. Everton #003399, Inter #010E3D).
      - Pure white / very near-white (luminance ≥ 0.80) → replaced with a darker
        palette entry. Yellow/gold kits (e.g. BVB, Wolves) are kept as-is because
        they ARE clearly visible on dark backgrounds; only text ON those bars needs
        to use dark ink (handled separately by _text_on_color).
    """
    r, g, b = _hex_to_rgb01(hex_color)
    lum = _relative_luminance(hex_color)
    max_ch = max(r, g, b)
    is_too_dark = (lum < 0.04 and max_ch < 0.18) or (lum < 0.10 and max_ch < 0.22)

    if is_too_dark or _contrast_ratio_hex(hex_color, BG_DARK) < 1.65:
        pal = _team_palette(team_name, fallback)
        return _readable_kit_candidate(
            team_name, pal[1:] + pal[:1], fallback, allow_light=False
        )

    # Pure white / off-white kits only (keep yellows/golds as visible bar colours)
    if lum >= 0.80:
        pal = _team_palette(team_name, fallback)
        return _readable_kit_candidate(
            team_name, pal[1:] + pal[:1], fallback, allow_light=False
        )

    return hex_color


def _kit_palette_index(kit_type):
    """Map configured kit type to a palette index."""
    k = str(kit_type or "auto").strip().lower()
    if k in {"", "auto", "smart"}:
        return None
    if k in {"home", "primary", "first"}:
        return 0
    if k in {"accent", "trim", "stripe", "second"}:
        return 1
    if k in {
        "away",
        "third",
        "alternate",
        "alt",
        "clash",
        "change",
        "second_kit",
        "third_kit",
    }:
        return 2
    return None


def _configured_kit_colour(team_name, kit_type, fallback):
    """Return the colour requested by HOME_KIT_TYPE / AWAY_KIT_TYPE."""
    idx = _kit_palette_index(kit_type)
    if idx is None:
        return None
    pal = _team_palette(team_name, fallback)
    idx = min(idx, len(pal) - 1)
    return _visible_on_dark(team_name, pal[idx], fallback)


def choose_matchup_colors(
    home_name: str, away_name: str, home_kit_type=None, away_kit_type=None
):
    """
    Pick kit-based colours for a match while keeping the home team's identity stable.

    v6 fix:
      - White/off-white primary kits are replaced with the team's accent or
        alternate colour so visuals remain readable on the dark background.
      - Home team uses a visible version of their HOME kit colour.
      - Away team NEVER uses pure white (#FFFFFF or near-white lum ≥ 0.82) as its
        display colour — white lines/text are invisible on bright chart elements.
        Instead the away team's best non-white palette entry is used.
      - Falls back to the primary if the alternate provides poor contrast.
      - Very light alternates are avoided when a readable non-light alternate is
        available with enough contrast.
    """
    custom_home = (CUSTOM_KIT_COLORS or {}).get("home")
    custom_away = (CUSTOM_KIT_COLORS or {}).get("away")
    forced_home = custom_home or _configured_kit_colour(
        home_name, home_kit_type, "#B91C1C"
    )
    forced_away = custom_away or _configured_kit_colour(
        away_name, away_kit_type, "#9CA3AF"
    )

    if custom_home and custom_away:
        return _visible_on_dark(home_name, forced_home, "#B91C1C"), _visible_on_dark(
            away_name, forced_away, "#9CA3AF"
        )

    home_palette_raw = _team_palette(home_name, DEFAULT_HOME)
    away_palette_raw = _team_palette(away_name, DEFAULT_AWAY)

    home_palette = [_usable_on_dark(c, "#B91C1C") for c in home_palette_raw]
    away_palette = [_usable_on_dark(c, "#9CA3AF") for c in away_palette_raw]

    # For the primary display colour, also avoid white/off-white on dark backgrounds
    home_primary = forced_home or _visible_on_dark(
        home_name, home_palette_raw[0], "#B91C1C"
    )
    away_primary = forced_away or _visible_on_dark(
        away_name, away_palette_raw[0], "#9CA3AF"
    )

    # ── Hard filter: never allow white/near-white as away colour ─────────────
    # White lines and text are invisible on legend boxes, stat labels and light
    # chart areas. If away_primary ended up white, force the next palette entry.
    def _is_near_white(col: str) -> bool:
        return _relative_luminance(col) >= 0.82

    def _best_non_white(palette_raw: list, primary: str, fb: str) -> str:
        """Return the best non-white visible entry from the palette."""
        if not _is_near_white(primary):
            return primary
        for c in palette_raw:
            cv = _visible_on_dark("", c, fb)
            if cv and not _is_near_white(cv) and _relative_luminance(cv) > 0.02:
                return cv
        return fb  # last resort

    away_primary = _best_non_white(away_palette_raw, away_primary, "#9CA3AF")

    # ── Away/alternate kit colour = best non-white entry from palette ─────────
    # Start from the 3rd palette entry (clash/away colour) and walk back.
    away_alternate_raw = (
        away_palette[-1]
        if len(away_palette) >= 3
        else (away_palette[1] if len(away_palette) >= 2 else away_primary)
    )
    away_alternate = _best_non_white(away_palette_raw, away_alternate_raw, away_primary)

    # Stronger light penalty — white-like colours are heavily discouraged
    def _light_penalty(col: str) -> float:
        lum = _relative_luminance(col)
        if lum >= 0.82:
            return 0.60  # near-white: almost disqualified
        if lum >= 0.76:
            return 0.25  # off-white: strong penalty
        return 0.0

    explicit_away_kit = _kit_palette_index(away_kit_type) == 2

    # ── Step 1: Home primary + Away alternate (explicit away kit only) ───────
    if explicit_away_kit and away_alternate != away_primary:
        alt_score = _color_distance(home_primary, away_alternate) - _light_penalty(
            away_alternate
        )
        pri_score = _color_distance(home_primary, away_primary) - _light_penalty(
            away_primary
        )
        if (
            _color_distance(home_primary, away_alternate) >= 0.28
            and alt_score >= pri_score * 0.85
        ):
            return home_primary, away_alternate

    # ── Step 2: Home primary + Away primary if contrast is OK ──────
    if _color_distance(home_primary, away_primary) >= 0.34:
        return home_primary, away_primary

    # ── Step 3: Try away palette colours for better contrast ───────
    best_away = away_primary
    best_score = _color_distance(home_primary, away_primary) - _light_penalty(
        away_primary
    )
    # Prefer alternate/away colours first, then the rest
    away_candidates = []
    if len(away_palette) >= 3:
        away_candidates.append(away_palette[2])  # alternate/away kit
    if len(away_palette) >= 2:
        away_candidates.append(away_palette[1])  # accent/stripe
    away_candidates += away_palette

    for ac in away_candidates:
        ac = _usable_on_dark(ac, "#9CA3AF")
        score = _color_distance(home_primary, ac) - _light_penalty(ac)
        if score > best_score:
            best_away, best_score = ac, score

    if _color_distance(home_primary, best_away) >= 0.28:
        return home_primary, best_away

    for ac in ["#9CA3AF", "#00A3E0", "#FDE100"]:
        ac = _usable_on_dark(ac, "#9CA3AF")
        score = _color_distance(home_primary, ac) - _light_penalty(ac)
        if score > best_score:
            best_away, best_score = ac, score

    if _color_distance(home_primary, best_away) >= 0.28:
        return home_primary, best_away

    # ── Step 4: Last resort — full palette search ──────────────────
    best = (
        home_primary,
        best_away,
        _color_distance(home_primary, best_away) - _light_penalty(best_away),
    )
    for hc in home_palette:
        hc = _usable_on_dark(hc, "#B91C1C")
        home_switch_penalty = 0.22 if hc != home_primary else 0.0
        for ac in away_palette + ["#9CA3AF", "#00A3E0", "#FDE100"]:
            ac = _usable_on_dark(ac, "#9CA3AF")
            score = _color_distance(hc, ac) - _light_penalty(ac) - home_switch_penalty
            if score > best[2]:
                best = (hc, ac, score)

    # Final safety: if still ended up with white/near-white away colour, force-replace
    final_home, final_away = best[0], best[1]
    if _is_near_white(final_away):
        final_away = _best_non_white(away_palette_raw, away_primary, "#9CA3AF")

    return final_home, final_away


HOME_COLOR = DEFAULT_HOME
AWAY_COLOR = DEFAULT_AWAY

BG_DARK = "#000000"
BG_MID = "#020202"
PITCH_COL = "#030303"
GRID_COL = "#2A2A2A"
TEXT_MAIN = "#F4F8FF"
TEXT_DIM = "#D6DEE8"
TEXT_BRIGHT = "#FFFFFF"


def _relative_luminance_hex(color: str) -> float:
    try:
        import matplotlib.colors as _mcolors

        r, g, b = _mcolors.to_rgb(color)
    except Exception:
        r, g, b = (1.0, 1.0, 1.0)
    vals = []
    for c in (r, g, b):
        vals.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def _contrast_ratio(fg: str, bg: str = BG_MID) -> float:
    l1, l2 = _relative_luminance_hex(fg), _relative_luminance_hex(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def readable_text_color(
    color: str, bg: str = BG_MID, fallback: str = TEXT_BRIGHT, min_ratio: float = 5.0
) -> str:
    try:
        return color if _contrast_ratio(color, bg) >= min_ratio else fallback
    except Exception:
        return fallback


def readable_team_text_color(color: str, bg: str = BG_MID) -> str:
    return readable_text_color(color, bg, fallback=TEXT_BRIGHT, min_ratio=5.0)


ACCENT_TEXT = readable_text_color(C_GOLD, BG_MID, fallback=TEXT_BRIGHT)


# Global AMOLED defaults: every legacy figure/axis starts from a near-black
# surface and high-contrast text, even if an individual visual forgets to set it.
def _apply_amoled_matplotlib_defaults() -> None:
    try:
        plt.rcParams.update(
            {
                "figure.facecolor": BG_DARK,
                "axes.facecolor": BG_MID,
                "savefig.facecolor": BG_DARK,
                "savefig.edgecolor": BG_DARK,
                "text.color": TEXT_MAIN,
                "axes.labelcolor": TEXT_DIM,
                "xtick.color": TEXT_DIM,
                "ytick.color": TEXT_DIM,
                "axes.edgecolor": GRID_COL,
                "grid.color": GRID_COL,
                "legend.facecolor": BG_MID,
                "legend.edgecolor": GRID_COL,
                "patch.force_edgecolor": False,
            }
        )
    except Exception:
        pass


_apply_amoled_matplotlib_defaults()


TEXT_SHADOW = [pe.withStroke(linewidth=2.6, foreground="#000000")]
TEXT_SHADOW_STRONG = [pe.withStroke(linewidth=3.4, foreground="#000000")]


def text_shadow(strong: bool = False) -> list:
    """Reusable black outline for labels placed over charts, maps, or team colours."""
    return TEXT_SHADOW_STRONG if strong else TEXT_SHADOW


def _apply_neon_backdrop(fig):
    """Apply the shared teal/purple neon backdrop when available."""
    try:
        from visualization_design import _neon_backdrop

        _neon_backdrop(fig)
    except Exception:
        pass


COLOR_SUB_IN = C_GREEN
COLOR_SUB_OUT = C_GOLD
COLOR_RED_CARD = "#e63946"
COLOR_BOTH_SUB = "#a855f7"

FINAL_THIRD_X = 66.7
PENALTY_BOX_X = 83.5
PENALTY_BOX_Y1 = 21.1
PENALTY_BOX_Y2 = 78.9

SHOT_TYPES = {
    "Goal": "Goal",
    "SavedShot": "On Target",
    "MissedShots": "Off Target",
    "BlockedShot": "Blocked",
    "ShotOnPost": "Off Target",
}

SHOT_FAMILY = {
    "Goal": "On Target",
    "SavedShot": "On Target",
    "MissedShots": "Off Target",
    "BlockedShot": "Blocked",
    "ShotOnPost": "Off Target",
}

SHOT_STYLE_RAW = {
    "Goal": ("*", "#FFD700", "#ffffff", 520, 8, "Goal"),
    "SavedShot": ("o", "#00FF87", "#a7f3d0", 220, 6, "SavedShot"),
    "MissedShots": ("X", "#FF6B6B", "#fca5a5", 180, 5, "MissedShots"),
    "BlockedShot": ("s", "#FFE66D", "#fed7aa", 180, 5, "BlockedShot"),
    "ShotOnPost": ("D", "#A855F7", "#d8b4fe", 200, 6, "ShotOnPost"),
}

SHOT_BREAKDOWN_KEYS = ["shots", "post", "on_target", "off_target", "blocked"]
SHOT_BREAKDOWN_LABELS = [
    "Total Shots",
    "Woodwork",
    "Shots on target",
    "Shots off target",
    "Shots blocked",
]

SHOT_SUMMARY_KEYS = [
    "Total Shots",
    "On Target",
    "Off Target",
    "Blocked",
    "Woodwork",
    "Goals",
]

PERIOD_CODES = {
    "PreMatch": "pre",
    "FirstHalf": "1h",
    "HalfTime": "ht",
    "SecondHalf": "2h",
    "ExtraTimeFirstHalf": "et1",
    "ExtraTimeHalfTime": "etht",
    "ExtraTimeSecondHalf": "et2",
    "PenaltyShootout": "pso",
    "FullTime": "ft",
}
PERIOD_SPANS = [
    (0, 45, "1h", "1st Half", "#071507", C_GOLD),
    (45, 90, "2h", "2nd Half", "#070715", "#64748b"),
    (90, 105, "et1", "ET 1st", "#150715", "#a855f7"),
    (105, 120, "et2", "ET 2nd", "#151507", "#64748b"),
    (120, 145, "pso", "Penalties", "#150707", C_RED),
]
STOPPAGE_PERIODS = {"1h": 45, "2h": 90, "et1": 105, "et2": 120}


def _is_penalty_shootout_period(period_raw=None, period_code=None) -> bool:
    """Return True for post-extra-time penalty shootout periods."""
    raw = str(period_raw or "").strip().lower().replace(" ", "")
    code = str(period_code or "").strip().lower().replace(" ", "")
    return code in {"pso", "penaltyshootout", "shootout", "penalties"} or raw in {
        "pso",
        "penaltyshootout",
        "shootout",
        "penalties",
    }


def _is_penalty_shootout_row(row) -> bool:
    """Identify shootout rows from parsed event fields or raw qualifier names."""
    if _is_penalty_shootout_period(row.get("period"), row.get("period_code")):
        return True
    qnames = row.get("qualifier_names", [])
    if isinstance(qnames, str):
        qnames = [qnames]
    joined = " ".join(str(q) for q in qnames).lower().replace(" ", "")
    return "penaltyshootout" in joined or "shootout" in joined


STATUS_BADGE = {
    "ft": ("■ Full Time", "#64748b"),
    "pso": ("■ Penalties FT", C_RED),
    "1h": ("● 1st Half", C_GREEN),
    "2h": ("● 2nd Half", C_GREEN),
    "et1": ("● ET 1st", "#a855f7"),
    "et2": ("● ET 2nd", "#a855f7"),
    "ht": ("◐ Half Time", C_GOLD),
    "etht": ("◐ ET Half Time", C_GOLD),
}

OG_COLOR = "#ff00ff"
OG_LABEL = "🔄 OG"


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def _short(name: str) -> str:
    if not name:
        return ""
    parts = name.strip().split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else name


def get_status(md: dict) -> str:
    status = md.get("matchHeader", {}).get("status", "")
    if status:
        return PERIOD_CODES.get(status, status.lower())
    evts = md.get("events", [])
    if evts:
        lp = evts[-1].get("period", {})
        if isinstance(lp, dict):
            return PERIOD_CODES.get(lp.get("displayName", ""), "ft")
    return "ft"


def _extract_match_data(html: str) -> dict:
    """
    Extract matchCentreData from HTML regardless of its source.
    Use brace counting to extract the complete JSON object.
    """
    soup = BeautifulSoup(html, "html.parser")
    script = soup.select_one('script:-soup-contains("matchCentreData")')
    if not script or not script.string:
        raise ValueError("matchCentreData not found in page HTML")

    raw = script.string
    marker = "matchCentreData: "
    idx = raw.find(marker)
    if idx == -1:
        raise ValueError("matchCentreData marker not found in script")

    start = raw.find("{", idx + len(marker))
    if start == -1:
        raise ValueError("JSON object start '{' not found")

    depth = 0
    in_str = False
    escape = False
    end = start

    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    jstr = raw[start:end]
    return json.loads(jstr)


def _capture_scraped_page(html: str, visible_text: str | None = None) -> None:
    global LAST_PAGE_HTML, LAST_PAGE_TEXT
    LAST_PAGE_HTML = html or ""
    if visible_text is None:
        try:
            visible_text = BeautifulSoup(LAST_PAGE_HTML, "html.parser").get_text(
                " ", strip=True
            )
        except Exception:
            visible_text = LAST_PAGE_HTML or ""
    LAST_PAGE_TEXT = visible_text or ""


def _patch_undetected_chromedriver_del(uc_module) -> None:
    """
    Suppress harmless Windows shutdown noise in undetected_chromedriver.

    Sometimes uc.Chrome.__del__ runs after Chrome/ChromeDriver has already
    closed its Windows handle, so the library prints:
        Exception ignored in: <function Chrome.__del__ ...>
        OSError: [WinError 6] The handle is invalid

    Patching __del__ alone is not always enough when uc creates a partially
    initialized Chrome object during a failed session start. Therefore we patch
    BOTH Chrome.quit() and Chrome.__del__ before every uc.Chrome(...) call.
    """
    try:
        chrome_cls = getattr(uc_module, "Chrome", None)
        if chrome_cls is None or getattr(chrome_cls, "_ws_safe_cleanup_patched", False):
            return

        original_quit = getattr(chrome_cls, "quit", None)

        def _safe_quit(self, *args, **kwargs):
            if original_quit is None:
                return None
            try:
                return original_quit(self, *args, **kwargs)
            except OSError:
                return None
            except Exception:
                return None

        def _safe_del(self):
            try:
                _safe_quit(self)
            except Exception:
                pass

        chrome_cls.quit = _safe_quit
        chrome_cls.__del__ = _safe_del
        chrome_cls._ws_safe_cleanup_patched = True
    except Exception:
        pass


def _safe_quit_driver(driver) -> None:
    """Close Selenium/Chrome drivers without printing noisy cleanup errors."""
    if driver is None:
        return
    try:
        driver.quit()
    except OSError:
        pass
    except Exception:
        pass


def _detect_chrome_major_version() -> int | None:
    """
    Detect the installed Google Chrome major version on Windows.

    undetected_chromedriver sometimes auto-downloads the newest driver instead
    of the driver matching the installed browser. Passing version_main=<major>
    prevents errors like:
      This version of ChromeDriver only supports Chrome version 148
      Current browser version is 147.x

    You can override detection manually by setting:
      set CHROME_VERSION_MAIN=147
    """
    env_val = os.environ.get("CHROME_VERSION_MAIN") or os.environ.get("UC_VERSION_MAIN")
    if env_val:
        m = re.search(r"\d+", str(env_val))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass

    # 1) Windows registry is the most reliable when Chrome is installed normally.
    try:
        import winreg  # type: ignore

        registry_locations = [
            (winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\Google\Chrome\BLBeacon"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Google\Chrome\BLBeacon"),
        ]
        for hive, key_path in registry_locations:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    version, _ = winreg.QueryValueEx(key, "version")
                m = re.search(r"^(\d+)", str(version))
                if m:
                    return int(m.group(1))
            except Exception:
                continue
    except Exception:
        pass

    # 2) Fallback: ask chrome.exe for its version.
    try:
        import subprocess

        candidate_paths = []
        for base in (
            os.environ.get("LOCALAPPDATA"),
            os.environ.get("PROGRAMFILES"),
            os.environ.get("PROGRAMFILES(X86)"),
        ):
            if base:
                candidate_paths.append(
                    os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
                )

        for exe_path in candidate_paths:
            if not exe_path or not os.path.exists(exe_path):
                continue
            try:
                out = subprocess.check_output(
                    [exe_path, "--version"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=5,
                )
                m = re.search(r"(\d+)\.\d+\.\d+\.\d+", out or "")
                if m:
                    return int(m.group(1))
            except Exception:
                continue
    except Exception:
        pass

    return None


def _uc_chrome_kwargs(opts, chromedriver_path: str | None = None) -> dict:
    """
    Build safe kwargs for uc.Chrome(), forcing the Chrome major version when it
    can be detected. This avoids uc downloading a mismatched future driver.
    """
    kw = {"options": opts}
    chrome_major = _detect_chrome_major_version()
    if chrome_major:
        kw["version_main"] = chrome_major
        try:
            console.print(f"[dim]  Detected Chrome major version: {chrome_major}[/dim]")
        except Exception:
            pass
    else:
        try:
            console.print(
                "[dim]  Chrome major version not detected; using uc default[/dim]"
            )
        except Exception:
            pass
    if chromedriver_path and os.path.exists(chromedriver_path):
        kw["driver_executable_path"] = chromedriver_path
    return kw


# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════


_HEADERS_POOL = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/142.0.7444.176 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Ch-Ua": '"Chromium";v="142", "Google Chrome";v="142"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
            "Gecko/20100101 Firefox/124.0"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.whoscored.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    },
]


def _try_cloudscraper(url: str) -> dict:
    """
    Attempt 1: use cloudscraper to bypass Cloudflare without a browser.
    pip install cloudscraper
    """
    try:
        import cloudscraper
    except ImportError:
        raise RuntimeError(
            "cloudscraper is not installed; run: pip install cloudscraper"
        )

    console.print("[cyan]  [1/3] Trying cloudscraper...[/cyan]")
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )

    scraper.get("https://www.whoscored.com/", timeout=30)
    time.sleep(random.uniform(2, 4))

    resp = scraper.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    if "matchCentreData" not in resp.text:
        raise RuntimeError("matchCentreData not found via cloudscraper")

    _capture_scraped_page(resp.text)
    console.print("[green]  cloudscraper succeeded![/green]")
    return _extract_match_data(resp.text)


def _try_requests(url: str) -> dict:
    """
    Attempt 2: use requests with a session, rotating headers, and cookies.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    console.print("[cyan]  [2/3] Trying requests + session...[/cyan]")

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))

    headers = random.choice(_HEADERS_POOL)
    session.headers.update(headers)

    session.get("https://www.whoscored.com/", timeout=30)
    time.sleep(random.uniform(3, 6))

    resp = session.get(url, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    if "matchCentreData" not in resp.text:
        raise RuntimeError("matchCentreData not found via requests")

    _capture_scraped_page(resp.text)
    console.print("[green]  requests succeeded![/green]")
    return _extract_match_data(resp.text)


def _try_chrome(
    url: str,
    chromedriver_path: str = None,
    profile_dir: str = None,
    profile_name: str = "Default",
) -> dict:
    """
    Attempt 3: use undetected_chromedriver with:
      ✅ A real Chrome profile for cookies and authentication
      ✅ selenium-stealth to conceal automation indicators
      ✅ Random delays that imitate human behavior
    """
    try:
        import undetected_chromedriver as uc

        _patch_undetected_chromedriver_del(uc)
    except ImportError:
        raise RuntimeError(
            "undetected_chromedriver is not installed; "
            "run: pip install undetected-chromedriver"
        )

    from selenium.webdriver.support.ui import WebDriverWait

    console.print("[cyan]  [3/3] Trying Chrome + stealth...[/cyan]")

    opts = uc.ChromeOptions()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--lang=en-US")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.7444.176 Safari/537.36"
    )

    temp_user_data_dir = tempfile.mkdtemp(prefix="ws_uc_profile_")
    using_real_profile = bool(
        BROWSER_USE_REAL_PROFILE and profile_dir and os.path.isdir(profile_dir)
    )
    if using_real_profile:
        opts.add_argument(f"--user-data-dir={profile_dir}")
        opts.add_argument(f"--profile-directory={profile_name}")
        console.print(f"[yellow]  Using Chrome profile: {profile_name}[/yellow]")
    else:
        opts.add_argument(f"--user-data-dir={temp_user_data_dir}")
        opts.add_argument("--profile-directory=Default")
        console.print("[yellow]  Using isolated temporary Chrome profile[/yellow]")

    kw = _uc_chrome_kwargs(opts, chromedriver_path)

    driver = None
    try:
        driver = uc.Chrome(**kw)

        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']});
                window.chrome = {runtime: {}};
            """},
        )

        try:
            from selenium_stealth import stealth

            stealth(
                driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
            console.print("[green]  selenium-stealth applied[/green]")
        except ImportError:
            pass

        console.print("[cyan]  Visiting homepage first...[/cyan]")
        driver.get("https://www.whoscored.com/")
        time.sleep(random.uniform(3, 6))

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.3);")
        time.sleep(random.uniform(1, 2))
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(random.uniform(1, 2))

        console.print(f"[cyan]  Loading match page...[/cyan]")
        driver.get(url)

        WebDriverWait(driver, 120).until(lambda d: "matchCentreData" in d.page_source)

        from selenium.webdriver.common.by import By

        def _try_click_stats_view():
            phrases = ["statistics", "summary", "match centre", "match center", "stats"]
            xpath_tpl = (
                "//*[self::a or self::button or self::li or self::span or self::div]"
                "[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{p}')]"
            )
            for phrase in phrases:
                try:
                    elems = driver.find_elements(By.XPATH, xpath_tpl.format(p=phrase))
                except Exception:
                    elems = []
                for el in elems[:8]:
                    try:
                        driver.execute_script(
                            "arguments[0].scrollIntoView({block:'center'});", el
                        )
                        time.sleep(0.2)
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(0.8)
                    except Exception:
                        continue

        visible_text = ""
        for _ in range(20):
            _try_click_stats_view()
            try:
                driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight * 0.45);"
                )
            except Exception:
                pass
            try:
                visible_text = (
                    driver.execute_script(
                        "return document.body ? document.body.innerText : ''; "
                    )
                    or ""
                )
            except Exception:
                visible_text = ""
            if re.search(
                r"Total\s+Team\s+xG|Shots\s+on\s+target|Shots\s+off\s+target|Shots\s+blocked|Woodwork",
                visible_text or "",
                flags=re.I,
            ):
                break
            time.sleep(1.0)

        html = driver.page_source
        _capture_scraped_page(html, visible_text)
        console.print("[green]  Chrome succeeded![/green]")
        return _extract_match_data(html)

    finally:
        _safe_quit_driver(driver)
        try:
            if (
                not using_real_profile
                and temp_user_data_dir
                and os.path.isdir(temp_user_data_dir)
            ):
                shutil.rmtree(temp_user_data_dir, ignore_errors=True)
        except Exception:
            pass


def scrape_match(
    url: str,
    chromedriver_path: str = None,
    profile_dir: str = None,
    profile_name: str = "Default",
) -> dict:
    """
    Try three methods in order and raise a clear exception if all fail.
    """
    errors = []

    try:
        return _try_cloudscraper(url)
    except Exception as e:
        msg = f"cloudscraper: {e}"
        errors.append(msg)
        console.print(f"[yellow]  ✗ {msg}[/yellow]")

    time.sleep(random.uniform(2, 4))

    try:
        return _try_requests(url)
    except Exception as e:
        msg = f"requests: {e}"
        errors.append(msg)
        console.print(f"[yellow]  ✗ {msg}[/yellow]")

    time.sleep(random.uniform(2, 4))

    try:
        return _try_chrome(url, chromedriver_path, profile_dir, profile_name)
    except Exception as e:
        msg = f"Chrome: {e}"
        errors.append(msg)
        console.print(f"[red]  ✗ {msg}[/red]")

    console.print("\n[bold red]═══ All attempts failed ═══[/bold red]")
    console.print("[yellow]Suggested solutions:[/yellow]")
    console.print("  1. Check your internet connection")
    console.print("  2. Run: pip install cloudscraper selenium-stealth")
    console.print("  3. Open whoscored.com in Chrome, sign in, and retry")
    console.print("  4. Use a VPN if the website is blocked in your region")
    raise RuntimeError(
        "Scraping failed with every available method:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ══════════════════════════════════════════════════════
#  xG MODEL
# ══════════════════════════════════════════════════════
GOAL_WIDTH, PITCH_LEN, PITCH_WID = 7.32, 105.0, 68.0
BODY_PART_IDS = {"foot": 0, "head": 1, "other": 2}
RESULT_IDS = {
    "fail": 0,
    "success": 1,
    "offside": 2,
    "owngoal": 3,
    "yellow_card": 4,
    "red_card": 5,
}
ACTION_TYPE_IDS = {
    "pass": 0,
    "cross": 1,
    "throw_in": 2,
    "freekick_crossed": 3,
    "freekick_short": 4,
    "corner_crossed": 5,
    "corner_short": 6,
    "take_on": 7,
    "foul": 8,
    "tackle": 9,
    "interception": 10,
    "shot": 11,
    "shot_penalty": 12,
    "shot_freekick": 13,
    "keeper_save": 14,
    "keeper_claim": 15,
    "keeper_punch": 16,
    "keeper_pick_up": 17,
    "clearance": 18,
    "bad_touch": 19,
    "non_action": 20,
    "dribble": 21,
    "goalkick": 22,
}
OPENPLAY_ADVANCED_COLUMNS = [
    "bodypart_id_a0",
    "start_dist_to_goal_a0",
    "start_angle_to_goal_a0",
    "type_id_a1",
    "type_id_a2",
    "bodypart_id_a1",
    "bodypart_id_a2",
    "result_id_a1",
    "result_id_a2",
    "start_x_a0",
    "start_y_a0",
    "start_x_a1",
    "start_y_a1",
    "start_x_a2",
    "start_y_a2",
    "end_x_a1",
    "end_y_a1",
    "end_x_a2",
    "end_y_a2",
    "dx_a1",
    "dy_a1",
    "movement_a1",
    "dx_a2",
    "dy_a2",
    "movement_a2",
    "dx_a01",
    "dy_a01",
    "mov_a01",
    "dx_a02",
    "dy_a02",
    "mov_a02",
    "start_dist_to_goal_a1",
    "start_angle_to_goal_a1",
    "start_dist_to_goal_a2",
    "start_angle_to_goal_a2",
    "end_dist_to_goal_a1",
    "end_angle_to_goal_a1",
    "end_dist_to_goal_a2",
    "end_angle_to_goal_a2",
    "time_delta_1",
    "time_delta_2",
    "speedx_a01",
    "speedy_a01",
    "speed_a01",
    "speedx_a02",
    "speedy_a02",
    "speed_a02",
    "shot_angle_a0",
    "shot_angle_a1",
    "shot_angle_a2",
]
FREEKICK_COLUMNS = ["start_dist_to_goal_a0", "start_angle_to_goal_a0"]
XG_MODEL_USED = "provider_or_opta_like_v4"


def _qnames(row_or_event) -> set[str]:
    quals = []
    if hasattr(row_or_event, "get"):
        quals = row_or_event.get("qualifiers", []) or []
    names = set()
    if isinstance(quals, (list, tuple)):
        for q in quals:
            if isinstance(q, dict):
                nm = q.get("type", {}).get("displayName", "")
                if nm:
                    names.add(str(nm))
    if names:
        return names

    q = row_or_event.get("qualifier_names", []) if hasattr(row_or_event, "get") else []
    if isinstance(q, str):
        return {x for x in q.split("|") if x}
    if isinstance(q, (list, tuple, set)):
        return {str(x) for x in q if x and str(x).lower() != "nan"}
    return set()


def _safe_float(v, default=0.0):
    try:
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _to_mx(x):
    return _clamp(_safe_float(x) * PITCH_LEN / 100.0, 0.0, PITCH_LEN)


def _to_my(y):
    return _clamp(_safe_float(y) * PITCH_WID / 100.0, 0.0, PITCH_WID)


def _period_id(period_code: str) -> int:
    return {"1h": 1, "2h": 2, "et1": 3, "et2": 4, "pso": 5}.get(period_code, 1)


def _overall_seconds(minute, second):
    return _safe_float(minute, 0.0) * 60.0 + _safe_float(second, 0.0)


def _bodypart_id_from_row(row) -> int:
    raw_body = row.get("body_part", "") if hasattr(row, "get") else ""
    if raw_body is None:
        body = ""
    elif isinstance(raw_body, str):
        body = raw_body.strip().lower()
    else:
        try:
            body = "" if pd.isna(raw_body) else str(raw_body).strip().lower()
        except Exception:
            body = str(raw_body).strip().lower()

    q = _qnames(row)
    if body in {"head", "header"} or "head" in body or "Head" in q:
        return BODY_PART_IDS["head"]
    if body in {"rightfoot", "leftfoot", "foot", "right foot", "left foot"}:
        return BODY_PART_IDS["foot"]
    if {"RightFoot", "LeftFoot"} & q:
        return BODY_PART_IDS["foot"]
    return BODY_PART_IDS["other"]


def _is_direct_freekick_row(row) -> bool:
    q = _qnames(row)
    return bool({"DirectFreekick", "FreekickShot", "Direct free kick"} & q) or bool(
        row.get("is_direct_fk", False)
    )


def _who_to_spadl_type(row) -> int | None:
    etype = row.get("type")
    q = _qnames(row)
    if etype in {
        "Goal",
        "SavedShot",
        "MissedShots",
        "BlockedShot",
        "ShotOnPost",
        "OwnGoal",
    } or row.get("is_shot"):
        if row.get("is_penalty") or ("Penalty" in q):
            return ACTION_TYPE_IDS["shot_penalty"]
        if _is_direct_freekick_row(row):
            return ACTION_TYPE_IDS["shot_freekick"]
        return ACTION_TYPE_IDS["shot"]
    if etype in {"Pass", "OffsidePass", "OffsidPass", "KeyPass"}:
        if "ThrowIn" in q:
            return ACTION_TYPE_IDS["throw_in"]
        if "GoalKick" in q:
            return ACTION_TYPE_IDS["goalkick"]
        if "CornerTaken" in q:
            return (
                ACTION_TYPE_IDS["corner_crossed"]
                if row.get("is_cross")
                else ACTION_TYPE_IDS["corner_short"]
            )
        if {"FreekickTaken", "SetPiece"} & q:
            return (
                ACTION_TYPE_IDS["freekick_crossed"]
                if row.get("is_cross")
                else ACTION_TYPE_IDS["freekick_short"]
            )
        return (
            ACTION_TYPE_IDS["cross"] if row.get("is_cross") else ACTION_TYPE_IDS["pass"]
        )
    if etype in {"TakeOn"}:
        return ACTION_TYPE_IDS["take_on"]
    if etype in {"Dribble"}:
        return ACTION_TYPE_IDS["dribble"]
    if etype in {"Foul", "FoulGiven", "FoulCommitted"}:
        return ACTION_TYPE_IDS["foul"]
    if etype in {"Tackle"}:
        return ACTION_TYPE_IDS["tackle"]
    if etype in {"Interception"}:
        return ACTION_TYPE_IDS["interception"]
    if etype in {"Clearance"}:
        return ACTION_TYPE_IDS["clearance"]
    if etype in {"KeeperSave"}:
        return ACTION_TYPE_IDS["keeper_save"]
    if etype in {"KeeperClaim"}:
        return ACTION_TYPE_IDS["keeper_claim"]
    if etype in {"KeeperPunch"}:
        return ACTION_TYPE_IDS["keeper_punch"]
    if etype in {"KeeperPickup", "KeeperPickUp"}:
        return ACTION_TYPE_IDS["keeper_pick_up"]
    if etype in {"BallTouch", "Dispossessed", "Error"}:
        return ACTION_TYPE_IDS["bad_touch"]
    return None


def _result_id_from_row(row) -> int:
    etype = row.get("type")
    outcome = row.get("outcome")
    if row.get("is_own_goal"):
        return RESULT_IDS["owngoal"]
    if etype == "Card":
        if "Red" in _qnames(row):
            return RESULT_IDS["red_card"]
        return RESULT_IDS["yellow_card"]
    if row.get("is_goal") or etype == "Goal":
        return RESULT_IDS["success"]
    if outcome == "Successful":
        return RESULT_IDS["success"]
    if outcome == "Offside":
        return RESULT_IDS["offside"]
    return RESULT_IDS["fail"]


def _flip_lr(df: pd.DataFrame, away_mask: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for col in [
        c for c in out.columns if c.startswith("start_x") or c.startswith("end_x")
    ]:
        out.loc[away_mask, col] = PITCH_LEN - out.loc[away_mask, col].values
    for col in [
        c for c in out.columns if c.startswith("start_y") or c.startswith("end_y")
    ]:
        out.loc[away_mask, col] = PITCH_WID - out.loc[away_mask, col].values
    return out


def _polar_dist_angle(x_ser, y_ser, prefix: str) -> pd.DataFrame:
    dx = (PITCH_LEN - x_ser).abs()
    dy = (PITCH_WID / 2.0 - y_ser).abs()
    out = pd.DataFrame(index=x_ser.index)
    out[f"{prefix}_dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[f"{prefix}_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return out


def _goal_angle_series(x_ser, y_ser) -> pd.Series:
    dx = PITCH_LEN - x_ser
    dy = PITCH_WID / 2.0 - y_ser
    denom = dx**2 + dy**2 - (GOAL_WIDTH / 2.0) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ang = np.arctan((GOAL_WIDTH * dx) / denom)
    ang = pd.Series(np.nan_to_num(ang), index=x_ser.index)
    ang.loc[ang < 0] += np.pi
    ang.loc[x_ser >= PITCH_LEN] = 0.0
    on_line = (x_ser == PITCH_LEN) & y_ser.between(
        PITCH_WID / 2.0 - GOAL_WIDTH / 2.0, PITCH_WID / 2.0 + GOAL_WIDTH / 2.0
    )
    ang.loc[on_line] = np.pi
    return ang


def _build_spadl_like_actions(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    df = events.copy().reset_index().rename(columns={"index": "orig_event_index"})
    df["type_id"] = df.apply(_who_to_spadl_type, axis=1)
    df = df[df["type_id"].notna()].copy()
    if df.empty:
        return df
    df["type_id"] = df["type_id"].astype(int)
    df["bodypart_id"] = df.apply(_bodypart_id_from_row, axis=1).astype(int)
    df["result_id"] = df.apply(_result_id_from_row, axis=1).astype(int)
    df["period_id"] = df["period_code"].map(_period_id).fillna(1).astype(int)
    df["time_seconds"] = [
        _overall_seconds(m, s) for m, s in zip(df["minute"], df["second"])
    ]
    df["start_x"] = df["x"].apply(_to_mx)
    df["start_y"] = df["y"].apply(_to_my)
    df["end_x"] = df["end_x"].apply(_to_mx)
    df["end_y"] = df["end_y"].apply(_to_my)
    df = df.sort_values(
        ["minute", "second", "orig_event_index"], kind="stable"
    ).reset_index(drop=True)
    return df


def _build_soccer_xg_feature_frame(
    actions: pd.DataFrame, home_team_id: int
) -> pd.DataFrame:
    if actions is None or actions.empty:
        return pd.DataFrame()

    def _prev(df, n):
        p = df.shift(n)
        if n > 0:
            p.iloc[:n] = df.iloc[[0] * n].values
        return p

    a0 = actions.copy()
    a1 = _prev(actions, 1)
    a2 = _prev(actions, 2)
    away_mask = a0["team_id"] != home_team_id
    a0 = _flip_lr(a0, away_mask)
    a1 = _flip_lr(a1, away_mask)
    a2 = _flip_lr(a2, away_mask)

    X = pd.DataFrame(index=actions.index)
    X["bodypart_id_a0"] = a0["bodypart_id"].astype(int)
    X["type_id_a1"] = a1["type_id"].astype(int)
    X["type_id_a2"] = a2["type_id"].astype(int)
    X["bodypart_id_a1"] = a1["bodypart_id"].astype(int)
    X["bodypart_id_a2"] = a2["bodypart_id"].astype(int)
    X["result_id_a1"] = a1["result_id"].astype(int)
    X["result_id_a2"] = a2["result_id"].astype(int)

    for nm, df_ in [("a0", a0), ("a1", a1), ("a2", a2)]:
        X[f"start_x_{nm}"] = df_["start_x"]
        X[f"start_y_{nm}"] = df_["start_y"]
    for nm, df_ in [("a1", a1), ("a2", a2)]:
        X[f"end_x_{nm}"] = df_["end_x"]
        X[f"end_y_{nm}"] = df_["end_y"]
        X[f"dx_{nm}"] = df_["end_x"] - df_["start_x"]
        X[f"dy_{nm}"] = df_["end_y"] - df_["start_y"]
        X[f"movement_{nm}"] = np.sqrt(X[f"dx_{nm}"] ** 2 + X[f"dy_{nm}"] ** 2)

    X["dx_a01"] = a1["end_x"] - a0["start_x"]
    X["dy_a01"] = a1["end_y"] - a0["start_y"]
    X["mov_a01"] = np.sqrt(X["dx_a01"] ** 2 + X["dy_a01"] ** 2)
    X["dx_a02"] = a2["end_x"] - a0["start_x"]
    X["dy_a02"] = a2["end_y"] - a0["start_y"]
    X["mov_a02"] = np.sqrt(X["dx_a02"] ** 2 + X["dy_a02"] ** 2)

    for nm, df_ in [("a0", a0), ("a1", a1), ("a2", a2)]:
        pol = _polar_dist_angle(df_["start_x"], df_["start_y"], "start")
        X[f"start_dist_to_goal_{nm}"] = pol["start_dist_to_goal"]
        X[f"start_angle_to_goal_{nm}"] = pol["start_angle_to_goal"]
        X[f"shot_angle_{nm}"] = _goal_angle_series(df_["start_x"], df_["start_y"])
    for nm, df_ in [("a1", a1), ("a2", a2)]:
        pol = _polar_dist_angle(df_["end_x"], df_["end_y"], "end")
        X[f"end_dist_to_goal_{nm}"] = pol["end_dist_to_goal"]
        X[f"end_angle_to_goal_{nm}"] = pol["end_angle_to_goal"]

    X["time_delta_1"] = (a0["time_seconds"] - a1["time_seconds"]).clip(lower=0)
    X["time_delta_2"] = (a0["time_seconds"] - a2["time_seconds"]).clip(lower=0)
    dt1 = X["time_delta_1"].replace(0, 1)
    dt2 = X["time_delta_2"].replace(0, 1)
    X["speedx_a01"] = X["dx_a01"].abs() / dt1
    X["speedy_a01"] = X["dy_a01"].abs() / dt1
    X["speed_a01"] = X["mov_a01"] / dt1
    X["speedx_a02"] = X["dx_a02"].abs() / dt2
    X["speedy_a02"] = X["dy_a02"].abs() / dt2
    X["speed_a02"] = X["mov_a02"] / dt2

    return X.replace([np.inf, -np.inf], 0).fillna(0)


def _sigmoid(z: float) -> float:
    z = _clamp(float(z), -12.0, 12.0)
    return 1.0 / (1.0 + math.exp(-z))


def _normalise_xg_value(v):
    """Return a valid 0..1 xG value, accepting percent-style values too."""
    if v is None:
        return None
    try:
        if isinstance(v, str):
            v = v.strip().replace("%", "")
            if not v:
                return None
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        # Some providers expose 7.9 instead of 0.079 or 79 instead of 0.79.
        if 1.0 < x <= 100.0:
            x = x / 100.0
        if 0.0 <= x <= 1.0:
            return round(float(_clamp(x, 0.001, XG_SINGLE_SHOT_CAP)), 4)
    except Exception:
        return None
    return None


def _key_looks_like_xg(key: str) -> bool:
    k = str(key or "").strip().lower().replace(" ", "").replace("_", "")
    if not k:
        return False
    # Do not confuse pitch x/endX coordinates with xG.
    if k in {"x", "y", "endx", "endy", "expandedminute"}:
        return False
    return (
        k
        in {
            "xg",
            "expectedgoals",
            "expectedgoal",
            "shotxg",
            "optaexpectedgoals",
            "optaxg",
        }
        or ("expected" in k and "goal" in k)
        or k.endswith("xg")
    )


def _extract_provider_shot_xg(row_or_event) -> float | None:
    """
    Use provider/Opta shot xG if WhoScored ever exposes it in the event payload.
    The current public matchCentreData often exposes only team-level xG, not
    shot-level xG, so this function is deliberately defensive and optional.
    """
    if not XG_USE_PROVIDER_SHOT_XG or not hasattr(row_or_event, "get"):
        return None

    # Direct columns/keys first.
    for key in (
        "xG",
        "xg",
        "expectedGoals",
        "expectedGoal",
        "expected_goals",
        "shotXG",
        "shot_xg",
        "optaXG",
        "opta_xg",
        "optaExpectedGoals",
    ):
        val = row_or_event.get(key)
        xg = _normalise_xg_value(val)
        if xg is not None:
            return xg

    # Qualifiers sometimes carry provider values.
    quals = row_or_event.get("qualifiers", []) or []
    if isinstance(quals, (list, tuple)):
        for q in quals:
            if not isinstance(q, dict):
                continue
            qtype = q.get("type", {}) if isinstance(q.get("type"), dict) else {}
            qname = (
                qtype.get("displayName") or q.get("displayName") or q.get("name") or ""
            )
            if _key_looks_like_xg(qname):
                for val_key in ("value", "displayValue", "numberValue"):
                    xg = _normalise_xg_value(q.get(val_key))
                    if xg is not None:
                        return xg

    # Controlled recursive scan through small nested objects only.
    def walk(obj, depth=0):
        if depth > 4:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _key_looks_like_xg(k):
                    xg = _normalise_xg_value(v)
                    if xg is not None:
                        return xg
                if isinstance(v, (dict, list, tuple)):
                    xg = walk(v, depth + 1)
                    if xg is not None:
                        return xg
        elif isinstance(obj, (list, tuple)):
            for item in list(obj)[:40]:
                xg = walk(item, depth + 1)
                if xg is not None:
                    return xg
        return None

    return walk(row_or_event)


def _context_flag(q: set[str], row, names: tuple[str, ...]) -> bool:
    lowered = {str(x).lower().replace(" ", "") for x in q}
    for nm in names:
        n = nm.lower().replace(" ", "")
        if n in lowered:
            return True
    for nm in names:
        val = row.get(nm) if hasattr(row, "get") else None
        if isinstance(val, bool) and val:
            return True
    return False


def _shot_geometry_features(row) -> dict:
    """Build the shot features used by the V7 internal xG engine."""
    q = _qnames(row)
    x_m = _to_mx(row.get("x"))
    y_m = _to_my(row.get("y"))
    dx = max(PITCH_LEN - x_m, 0.01)
    dy = abs(y_m - (PITCH_WID / 2.0))
    distance = math.hypot(dx, dy)
    denom = dx * dx + dy * dy - (GOAL_WIDTH / 2.0) ** 2
    angle = math.atan((GOAL_WIDTH * dx) / denom) if abs(denom) > 1e-9 else math.pi / 2
    if angle < 0:
        angle += math.pi
    angle = float(_clamp(angle, 0.0, math.pi))
    in_box = x_m >= (PITCH_LEN - 16.5)
    in_six = x_m >= (PITCH_LEN - 5.5) and dy <= 9.16
    central = max(0.0, 1.0 - dy / 34.0)
    central_box = in_box and dy <= 12.0
    return {
        "q": q,
        "x_m": x_m,
        "y_m": y_m,
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "angle": angle,
        "angle_norm": angle / math.pi,
        "in_box": in_box,
        "in_six": in_six,
        "central": central,
        "central_box": central_box,
        "wide": dy > 18.0,
    }


def _shot_context_features(row, f: dict | None = None) -> dict:
    """Provider-agnostic context flags for Opta/WhoScored shot events."""
    q = _qnames(row)
    return {
        "is_header": bool(row.get("is_header")) or ("Head" in q),
        "is_big": bool(row.get("big_chance")) or ("BigChance" in q),
        "is_direct_fk": _is_direct_freekick_row(row),
        "is_cross": bool(row.get("is_cross")) or ("Cross" in q),
        "is_fast": _context_flag(q, row, ("FastBreak", "CounterAttack")),
        "is_through": _context_flag(q, row, ("ThroughBall", "Through Ball")),
        "is_layoff": _context_flag(
            q, row, ("LayOff", "Layoff", "PullBack", "CutBack", "Pull Back", "Cut Back")
        ),
        "is_chipped": _context_flag(q, row, ("Chipped", "ChippedPass")),
        "is_set_piece": _context_flag(
            q, row, ("SetPiece", "FreekickTaken", "CornerTaken", "FromCorner")
        ),
        "is_rebound": _context_flag(
            q, row, ("Rebound", "SavedShot", "Blocked", "Save")
        ),
        "is_volley": _context_flag(q, row, ("Volley", "HalfVolley")),
        "is_one_on_one": _context_flag(q, row, ("OneOnOne", "One v One", "OneVsOne")),
    }


def _xg_foot_shot(f: dict, ctx: dict) -> float:
    """Open-play foot shot submodel (v3 — recalibrated).

    Based on published research:
    - Anzer & Bauer (2021): 105K Bundesliga shots, AUC=0.823
    - Iapteff et al. (2025): Bayesian xG, Frontiers in Sports
    - Robberechts & Davis (2020): KU Leuven, distance*angle interaction

    v3 changes vs v2: lower intercept and reduced context bonuses to prevent
    inflation when multiple flags stack.  Big_chance capped at 0.65 (was 1.05),
    in_six at 0.40 (was 0.60), in_box at 0.22 (was 0.35).  Intercept moved
    from -1.75 to -2.30 to anchor the baseline closer to Opta-scale.
    """
    d = f["distance"]
    a = f["angle"]
    c = f["central"]
    z = (
        -2.30
        - 0.108 * d
        + 1.82 * a
        + 0.0021 * d * d
        - 0.028 * (d * a)
        + 0.28 * c
        + (0.22 if f["in_box"] else 0.0)
        + (0.40 if f["in_six"] else 0.0)
        + (0.65 if ctx["is_big"] else 0.0)
        + (0.32 if ctx["is_through"] else 0.0)
        + (0.25 if ctx["is_layoff"] else 0.0)
        + (0.18 if ctx["is_rebound"] else 0.0)
        + (0.12 if ctx["is_fast"] else 0.0)
        + (0.45 if ctx["is_one_on_one"] else 0.0)
        - (0.24 if ctx["is_cross"] and not ctx["is_header"] else 0.0)
        - (0.18 if ctx["is_volley"] else 0.0)
        - (0.25 if ctx["is_set_piece"] else 0.0)
    )
    return _sigmoid(z)


def _xg_header_shot(f: dict, ctx: dict) -> float:
    """Header submodel (v3 — recalibrated).

    Separate model for headers: steeper distance penalty,
    different intercept, corner/cross context matters.

    v3 changes vs v2: intercept from -2.80 to -2.60, distance penalty 0.165
    (was 0.150), big_chance from 0.85 to 0.70, in_six from 0.80 to 0.70.
    This brings header xG in line with published values (typical header from
    8m ≈ 0.04, big-chance header from 6yd ≈ 0.22).
    """
    d = f["distance"]
    a = f["angle"]
    c = f["central"]
    z = (
        -2.60
        - 0.165 * d
        + 1.30 * a
        - 0.012 * (d * a)
        + 0.38 * c
        + (0.70 if f["in_six"] else 0.0)
        + (0.70 if ctx["is_big"] else 0.0)
        + (0.08 if ctx["is_cross"] else 0.0)
        + (0.20 if ctx["is_rebound"] else 0.0)
        - (0.10 if ctx["is_set_piece"] else 0.0)
    )
    return _sigmoid(z)


def _xg_direct_free_kick(f: dict) -> float:
    """Direct free kick submodel (v3 — recalibrated).

    Full logistic model instead of a lookup table.
    v3: reduced distance penalty (0.08 vs 0.165) because DFKs are taken
    from a dead ball with a wall — distance matters less than angle/centrality.
    Intercept from -2.10 to -1.80 so a 25m central DFK ≈ 0.06.
    """
    d = f["distance"]
    a = f["angle"]
    c = f["central"]
    z = -1.80 - 0.08 * d + 2.50 * a + 0.50 * c
    return _sigmoid(z)


# ── Removed: old ensemble and cap functions (v8.1) ──
# _xg_context_logit_bonus, _ml_logistic_xg_from_features, _cap_public_xg_value
# have been replaced by the three submodel functions above.


def _opta_like_local_xg_from_row(row) -> float:
    """
    xG v2 engine: Submodel architecture (foot / header / direct free kick / penalty).

    Research-backed logistic regression models:
    - Anzer & Bauer (2021): 105K shots, separate submodels by shot type
    - Iapteff et al. (2025): Bayesian xG, 7-feature model
    - Robberechts & Davis (2020): Distance×Angle interaction term
    - DataBallPy: Simple validated distance+angle models

    Pipeline:
      1. Penalty → 0.79 (world-average conversion rate)
      2. Direct free kick → _xg_direct_free_kick()
      3. Header → _xg_header_shot()
      4. Foot shot → _xg_foot_shot()
      5. Clamp result to [0.001, 0.95]
    """
    q = _qnames(row)
    provider_xg = _extract_provider_shot_xg(row)
    if provider_xg is not None:
        return provider_xg

    if row.get("is_penalty") or ("Penalty" in q):
        return XG_PENALTY_VALUE

    f = _shot_geometry_features(row)
    ctx = _shot_context_features(row, f)

    # Route to the appropriate submodel
    if ctx["is_direct_fk"]:
        xg = _xg_direct_free_kick(f)
    elif ctx["is_header"]:
        xg = _xg_header_shot(f, ctx)
    else:
        xg = _xg_foot_shot(f, ctx)

    # Simple bounds — no complex rule-based caps needed
    return round(float(_clamp(xg, 0.001, XG_SINGLE_SHOT_CAP)), 4)


def _open_event_xg_from_row(row) -> float:
    """Backward-compatible name; now routes to the v2 submodel xG engine."""
    return _opta_like_local_xg_from_row(row)


def apply_best_open_source_xg(events: pd.DataFrame, info: dict) -> pd.DataFrame:
    global XG_MODEL_USED
    if events is None or events.empty:
        return events

    out = events.copy()
    if "xG" not in out.columns:
        out["xG"] = np.nan
    out["xG"] = pd.to_numeric(out["xG"], errors="coerce")
    if "xg_source" not in out.columns:
        out["xg_source"] = ""

    shot_mask = out["is_shot"] == True
    if "is_own_goal" in out.columns:
        shot_mask &= ~out["is_own_goal"].fillna(False)
    if "is_penalty_shootout" in out.columns:
        shot_mask &= ~out["is_penalty_shootout"].fillna(False)
    if not shot_mask.any():
        return out

    if XG_USE_PROVIDER_SHOT_XG:
        provider_mask = (
            shot_mask & out["xG"].notna() & out["xG"].between(0.001, XG_SINGLE_SHOT_CAP)
        )
    else:
        provider_mask = pd.Series(False, index=out.index)
        out.loc[shot_mask, "xG"] = np.nan
    out.loc[provider_mask, "xg_source"] = "provider_shot_xg"

    local_mask = shot_mask & (~provider_mask)
    if local_mask.any():
        out.loc[local_mask, "xG"] = out.loc[local_mask].apply(
            _opta_like_local_xg_from_row, axis=1
        )
        out.loc[local_mask, "xg_source"] = XG_LOCAL_MODEL_VERSION

    XG_MODEL_USED = XG_LOCAL_MODEL_VERSION
    return out


def compute_xg(shot: dict) -> float:
    return _open_event_xg_from_row(shot)


def summarise_shots(events: list, team_id: int) -> dict:
    """
    Aggregate shot statistics using WhoScored's definitions:
      Goal + SavedShot           => On Target
      MissedShots + ShotOnPost   => Off Target
      BlockedShot                => Blocked
    Keep woodwork as a separate count as well.
    """
    summary = {k: 0 for k in SHOT_SUMMARY_KEYS}
    summary["xG"] = 0.0

    for ev in events:
        if ev.get("teamId") != team_id:
            continue
        raw_type = ev.get("type", {}).get("displayName", "")
        if raw_type not in SHOT_TYPES:
            continue

        summary["Total Shots"] += 1

        if raw_type == "Goal":
            summary["Goals"] += 1
            summary["On Target"] += 1
        elif raw_type == "SavedShot":
            summary["On Target"] += 1
        elif raw_type == "MissedShots":
            summary["Off Target"] += 1
        elif raw_type == "BlockedShot":
            summary["Blocked"] += 1
        elif raw_type == "ShotOnPost":
            summary["Off Target"] += 1
            summary["Woodwork"] += 1

        summary["xG"] += compute_xg(ev)

    summary["xG"] = round(summary["xG"], 2)
    summary["xG_per_shot"] = (
        round(summary["xG"] / summary["Total Shots"], 3)
        if summary["Total Shots"] > 0
        else 0.0
    )
    return summary


def calc_xg(
    x,
    y,
    header=False,
    penalty=False,
    big_chance=False,
    body_part=None,
    is_counter=False,
    is_direct_fk=False,
) -> float:
    shot = {
        "x": x,
        "y": y,
        "is_header": header,
        "is_penalty": penalty,
        "big_chance": big_chance,
        "body_part": body_part,
        "is_direct_fk": is_direct_fk,
        "qualifier_names": ["FastBreak"] if is_counter else [],
    }
    return compute_xg(shot)


# ══════════════════════════════════════════════════════
#  xT MODEL  (Karun Singh 12×8 grid)
#  Rows = 12 pitch-length zones (0→100), Cols = 8 pitch-width zones
# ══════════════════════════════════════════════════════
XT_GRID = np.array(
    [
        [0.00638, 0.00779, 0.00900, 0.00938, 0.00938, 0.00900, 0.00779, 0.00638],
        [0.00779, 0.01023, 0.01177, 0.01270, 0.01270, 0.01177, 0.01023, 0.00779],
        [0.00900, 0.01177, 0.01461, 0.01661, 0.01661, 0.01461, 0.01177, 0.00900],
        [0.01012, 0.01429, 0.01858, 0.02390, 0.02390, 0.01858, 0.01429, 0.01012],
        [0.01202, 0.01762, 0.02531, 0.03609, 0.03609, 0.02531, 0.01762, 0.01202],
        [0.01567, 0.02374, 0.03824, 0.06166, 0.06166, 0.03824, 0.02374, 0.01567],
        [0.02349, 0.03940, 0.07547, 0.14508, 0.14508, 0.07547, 0.03940, 0.02349],
        [0.03766, 0.07373, 0.16357, 0.40000, 0.40000, 0.16357, 0.07373, 0.03766],
        [0.05945, 0.12030, 0.26030, 0.54000, 0.54000, 0.26030, 0.12030, 0.05945],
        [0.09042, 0.18360, 0.36450, 0.62000, 0.62000, 0.36450, 0.18360, 0.09042],
        [0.12875, 0.25690, 0.46560, 0.70000, 0.70000, 0.46560, 0.25690, 0.12875],
        [0.17438, 0.33880, 0.56250, 0.76000, 0.76000, 0.56250, 0.33880, 0.17438],
    ]
)


def get_xt(x, y) -> float:
    """Look up xT value from Karun Singh 12×8 grid. Safe for any input."""
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(fx) or math.isnan(fy):
        return 0.0
    col = max(0, min(int(fx / 100.0 * 12), 11))
    row = max(0, min(int(fy / 100.0 * 8), 7))
    return float(XT_GRID[col, row])


def calc_xt_pass(x, y, end_x, end_y) -> float:
    """xT gained by a pass: destination xT minus origin xT. Returns 0 on bad input."""
    try:
        if any(v is None for v in [x, y, end_x, end_y]):
            return 0.0
        fx, fy = float(x), float(y)
        fex, fey = float(end_x), float(end_y)
        if any(math.isnan(v) for v in [fx, fy, fex, fey]):
            return 0.0
        return round(get_xt(fex, fey) - get_xt(fx, fy), 4)
    except (TypeError, ValueError):
        return 0.0


def has_q(quals, name: str) -> bool:
    return isinstance(quals, list) and any(
        q.get("type", {}).get("displayName") == name for q in quals
    )


def get_shot_family(raw_type: str) -> str | None:
    return SHOT_FAMILY.get(raw_type)


def get_shot_counts(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "shots": 0,
            "goals": 0,
            "saved": 0,
            "on_target": 0,
            "missed": 0,
            "off_target": 0,
            "blocked": 0,
            "post": 0,
        }

    raw = (
        df["shot_whoscored_type"]
        if "shot_whoscored_type" in df.columns
        else df["shot_category"].map(
            {
                "Goal": "Goal",
                "On Target": "SavedShot",
                "Off Target": "MissedShots",
                "Blocked": "BlockedShot",
                "Woodwork": "ShotOnPost",
            }
        )
    )

    goals = int(raw.eq("Goal").sum())
    saved = int(raw.eq("SavedShot").sum())
    missed = int(raw.eq("MissedShots").sum())
    blocked = int(raw.eq("BlockedShot").sum())
    post = int(raw.eq("ShotOnPost").sum())
    return {
        "shots": int(raw.notna().sum()),
        "goals": goals,
        "saved": saved,
        "on_target": goals + saved,
        "missed": missed + post,
        "off_target": missed + post,
        "blocked": blocked,
        "post": post,
    }


# ══════════════════════════════════════════════════════
#  OFFICIAL WHOSCORED STATS
# ══════════════════════════════════════════════════════
def _norm_team_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("manchester", "man")
    s = s.replace("utd", "united")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _parse_lr_stat(text: str, label_patterns: list[str], as_float: bool = False):
    cast = float if as_float else lambda x: int(round(float(x)))
    number = r"\d+(?:\.\d+)?"
    for label_re in label_patterns:
        patterns = [
            rf"(?P<h>{number})\s*{label_re}\s*(?P<a>{number})",
            rf"{label_re}\s*(?P<h>{number})\s*(?P<a>{number})",
            rf"(?P<h>{number})\s*(?P<a>{number})\s*{label_re}",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            try:
                return cast(m.group("h")), cast(m.group("a"))
            except Exception:
                continue
    return None, None


def _norm_stat_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key or "").strip().lower())


def _numeric_total(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            fv = float(value)
            return None if math.isnan(fv) else fv
        except Exception:
            return None
    if isinstance(value, dict):
        vals = []
        for v in value.values():
            num = _numeric_total(v)
            if num is not None:
                vals.append(num)
        return sum(vals) if vals else None
    if isinstance(value, (list, tuple)):
        vals = []
        for v in value:
            num = _numeric_total(v)
            if num is not None:
                vals.append(num)
        return sum(vals) if vals else None
    try:
        s = str(value).strip().replace("%", "")
        if not s:
            return None
        fv = float(s)
        return None if math.isnan(fv) else fv
    except Exception:
        return None


def _extract_official_from_flat_mapping(stats: dict) -> dict:
    if not isinstance(stats, dict):
        return {}
    out = {}
    aliases = {
        "xG": [
            "xg",
            "totalteamxg",
            "teamxg",
            "totalxg",
            "expectedgoals",
            "expectedgoal",
            "goalsexpected",
            "expectedgoalsfor",
            "shotxg",
            "shotsxg",
        ],
        "shots": ["shotstotal", "totalshots", "shots", "shotsattempted"],
        "on_target": ["shotsontarget", "shotontarget", "ontarget", "shotsongoal"],
        "off_target": ["shotsofftarget", "shotofftarget", "offtarget", "missedshots"],
        "blocked": ["shotsblocked", "blockedshots", "shotblocked", "blocked"],
        "woodwork": ["woodwork", "shotsonpost", "shotonpost", "hitwoodwork", "post"],
    }
    norm_map = {_norm_stat_key(k): k for k in stats.keys()}
    for out_key, keys in aliases.items():
        found = None
        for alias in keys:
            for nk, real_key in norm_map.items():
                if alias == nk or alias in nk:
                    found = _numeric_total(stats.get(real_key))
                    if found is not None:
                        break
            if found is not None:
                break
        if found is not None:
            out[out_key] = (
                round(float(found), 2) if out_key == "xG" else int(round(float(found)))
            )
    return out


def _coerce_js_like_literal(js_text: str) -> str:
    if not js_text:
        return js_text
    s = js_text
    s = re.sub(r"\bnull\b", "None", s)
    s = re.sub(r"\btrue\b", "True", s, flags=re.I)
    s = re.sub(r"\bfalse\b", "False", s, flags=re.I)
    return s


def _extract_js_block_after_marker(
    raw: str, marker: str, opener: str = "[", closer: str = "]"
) -> str:
    if not raw:
        return ""
    idx = raw.find(marker)
    if idx == -1:
        return ""
    start = raw.find(opener, idx + len(marker))
    if start == -1:
        return ""
    depth = 0
    in_str = False
    quote = ""
    escape = False
    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
                continue
            if ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return ""


def _collect_stat_pairs_from_node(node, out: dict):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (list, tuple, dict)):
                _collect_stat_pairs_from_node(v, out)
            else:
                nv = _numeric_total(v)
                if nv is not None and isinstance(k, str):
                    out[k] = nv
        return
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and isinstance(node[0], str):
            nums = []
            for x in node[1:]:
                nv = _numeric_total(x)
                if nv is not None:
                    nums.append(nv)
            if nums:
                out[node[0]] = nums[-1]
        for item in node:
            if isinstance(item, (list, tuple, dict)):
                _collect_stat_pairs_from_node(item, out)


def _extract_official_stats_from_initialdata(html: str) -> dict:
    if not html:
        return {}
    block = _extract_js_block_after_marker(html, "var initialData =", "[", "]")
    if not block:
        return {}
    try:
        data = ast.literal_eval(_coerce_js_like_literal(block))
    except Exception:
        return {}

    team_details = None
    try:
        if (
            isinstance(data, (list, tuple))
            and data
            and isinstance(data[0], (list, tuple))
            and len(data[0]) > 1
        ):
            team_details = data[0][1]
    except Exception:
        team_details = None
    if not isinstance(team_details, (list, tuple)) or len(team_details) < 2:
        return {}

    parsed = {"home": {}, "away": {}}
    for side, team in zip(("home", "away"), team_details[:2]):
        flat = {}
        try:
            stats_block = (
                team[3] if isinstance(team, (list, tuple)) and len(team) > 3 else None
            )
            _collect_stat_pairs_from_node(stats_block, flat)
        except Exception:
            flat = {}
        parsed[side] = _extract_official_from_flat_mapping(flat)
    parsed = _finalize_official_stats(parsed)
    return parsed if _official_stats_score(parsed) > 0 else {}


def _candidate_official_urls(url: str) -> list[str]:
    cands = []

    def add(u):
        if u and u not in cands:
            cands.append(u)

    add(url)
    add(re.sub(r"/live(/|$)", r"/livestatistics\1", url, flags=re.I))
    add(re.sub(r"/live(/|$)", r"/matchstatistics\1", url, flags=re.I))
    add(url.replace("/live/", "/LiveStatistics/"))
    add(url.replace("/Live/", "/LiveStatistics/"))
    return cands


def _fetch_html_via_http(url: str) -> tuple[str, str]:
    html = ""
    text = ""
    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        resp = scraper.get(url, timeout=60)
        if resp is not None and getattr(resp, "status_code", None) == 200:
            html = resp.text or ""
    except Exception:
        pass
    if not html:
        try:
            import requests

            headers = random.choice(_HEADERS_POOL)
            resp = requests.get(url, headers=headers, timeout=60)
            if resp is not None and getattr(resp, "status_code", None) == 200:
                html = resp.text or ""
        except Exception:
            pass
    if html:
        try:
            text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
        except Exception:
            text = html
    return html, text


def _extract_official_stats_http_only(base_url: str) -> dict:
    merged = {"home": {}, "away": {}}

    if LAST_PAGE_HTML or LAST_PAGE_TEXT:
        for candidate in (
            _extract_official_stats_from_initialdata(LAST_PAGE_HTML),
            _extract_official_stats_from_text(LAST_PAGE_TEXT),
            _extract_official_stats_from_text(LAST_PAGE_HTML),
        ):
            merged = _merge_official_stats(merged, candidate)
            if _official_stats_has(merged):
                return _finalize_official_stats(merged)

    for cand_url in _candidate_official_urls(base_url):
        console.print(f"[cyan]  Trying official stats via HTTP: {cand_url}[/cyan]")
        html, text = _fetch_html_via_http(cand_url)
        if not html and not text:
            continue
        for candidate in (
            _extract_official_stats_from_initialdata(html),
            _extract_official_stats_from_text(text),
            _extract_official_stats_from_text(html),
        ):
            merged = _merge_official_stats(merged, candidate)
            if _official_stats_has(merged):
                return _finalize_official_stats(merged)

    return _finalize_official_stats(merged) if _official_stats_score(merged) > 0 else {}


def _extract_matchcentre_team_stats(team_data: dict) -> dict:
    stats = (team_data or {}).get("stats", {}) or {}
    out = {}

    aliases = {
        "xG": [
            "xg",
            "totalteamxg",
            "teamxg",
            "totalxg",
            "expectedgoals",
            "expectedgoal",
            "goalsexpected",
            "expectedgoalsfor",
            "shotxg",
            "shotsxg",
        ],
        "shots": [
            "shotstotal",
            "totalshots",
            "shots",
            "shotsattempted",
        ],
        "on_target": [
            "shotsontarget",
            "shotontarget",
            "ontarget",
            "shotsongoal",
        ],
        "off_target": [
            "shotsofftarget",
            "shotofftarget",
            "offtarget",
            "missedshots",
        ],
        "blocked": [
            "shotsblocked",
            "blockedshots",
            "shotblocked",
            "blocked",
        ],
        "woodwork": [
            "woodwork",
            "shotsonpost",
            "shotonpost",
            "hitwoodwork",
            "post",
        ],
    }

    norm_map = {_norm_stat_key(k): k for k in stats.keys()}
    for out_key, keys in aliases.items():
        found = None
        for alias in keys:
            for nk, real_key in norm_map.items():
                if alias == nk or alias in nk:
                    found = _numeric_total(stats.get(real_key))
                    if found is not None:
                        break
            if found is not None:
                break
        if found is not None:
            out[out_key] = (
                round(float(found), 2) if out_key == "xG" else int(round(float(found)))
            )

    # Generic fallback for xG in case the provider uses a new key name
    if out.get("xG") is None:
        for real_key, raw_val in stats.items():
            nk = _norm_stat_key(real_key)
            if "xg" in nk or ("expected" in nk and "goal" in nk):
                found = _numeric_total(raw_val)
                if found is not None:
                    out["xG"] = round(float(found), 2)
                    break

    return out


def _extract_matchcentre_stats(md: dict) -> dict:
    if not isinstance(md, dict):
        return {}
    home = _extract_matchcentre_team_stats((md.get("home") or {}))
    away = _extract_matchcentre_team_stats((md.get("away") or {}))
    out = {"home": home, "away": away}
    return _finalize_official_stats(out) if home or away else {}


def _merge_official_stats(*stats_dicts: dict) -> dict:
    merged = {"home": {}, "away": {}}
    for stats in stats_dicts:
        if not isinstance(stats, dict):
            continue
        for side in ("home", "away"):
            vals = stats.get(side, {}) or {}
            if not isinstance(vals, dict):
                continue
            merged[side].update({k: v for k, v in vals.items() if v is not None})
    return merged


def _official_stats_score(stats: dict) -> int:
    score = 0
    for side in ("home", "away"):
        score += len((stats or {}).get(side, {}) or {})
    return score


def _official_stats_has(stats: dict, required=("xG", "shots", "on_target")) -> bool:
    if not isinstance(stats, dict):
        return False
    for side in ("home", "away"):
        vals = stats.get(side, {}) or {}
        if not all(k in vals for k in required):
            return False
    return True


def _finalize_official_stats(stats: dict) -> dict:
    out = {
        "home": dict((stats or {}).get("home", {}) or {}),
        "away": dict((stats or {}).get("away", {}) or {}),
    }
    for side in ("home", "away"):
        vals = out[side]
        shots = vals.get("shots")
        on_target = vals.get("on_target")
        off_target = vals.get("off_target")
        blocked = vals.get("blocked")
        woodwork = vals.get("woodwork")

        if (
            shots is not None
            and on_target is not None
            and off_target is not None
            and blocked is None
        ):
            vals["blocked"] = max(int(shots) - int(on_target) - int(off_target), 0)
        if (
            shots is not None
            and on_target is not None
            and blocked is not None
            and off_target is None
        ):
            vals["off_target"] = max(int(shots) - int(on_target) - int(blocked), 0)
        if (
            shots is not None
            and off_target is not None
            and blocked is not None
            and on_target is None
        ):
            vals["on_target"] = max(int(shots) - int(off_target) - int(blocked), 0)
        if vals.get("woodwork") is None:
            vals["woodwork"] = int(woodwork or 0)
        for key in ("shots", "on_target", "off_target", "blocked", "woodwork"):
            if vals.get(key) is not None:
                vals[key] = int(round(float(vals[key])))
        if vals.get("xG") is not None:
            vals["xG"] = round(float(vals["xG"]), 2)
    return out


def _strip_external_xg_totals(stats: dict) -> dict:
    """Remove provider/public team xG totals so V7 remains a fully internal xG model."""
    out = _finalize_official_stats(stats or {})
    for side in ("home", "away"):
        if side in out and isinstance(out[side], dict):
            out[side]["xG"] = None
    return out


def _extract_official_stats_from_text(text: str) -> dict:
    if not text:
        return {}

    raw = text.replace(" ", " ")
    # keep line structure for row-style stat parsing, but also build a flattened fallback string
    flat = re.sub(r"[|·•]+", " ", raw)
    flat = re.sub(r"\s+", " ", flat).strip()
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]

    labels = {
        "xG": ([r"Total\s+Team\s+xG", r"Team\s+xG", r"Total\s+xG"], True),
        "shots": ([r"Total\s+Shots", r"\bShots\b"], False),
        "on_target": ([r"Shots\s+on\s+target", r"On\s+Target"], False),
        "off_target": ([r"Shots\s+off\s+target", r"Off\s+Target"], False),
        "blocked": (
            [r"Shots\s+blocked", r"Blocked\s+Shots", r"Shots\s+Blocked"],
            False,
        ),
        "woodwork": ([r"Woodwork"], False),
    }

    out = {"home": {}, "away": {}}

    # 1) flattened regex pass
    for key, (patterns, is_float) in labels.items():
        h, a = _parse_lr_stat(flat, patterns, as_float=is_float)
        if h is not None and a is not None:
            out["home"][key] = h
            out["away"][key] = a

    # 2) line-aware pass: prev/current/next often looks like [left, label, right]
    number_re = r"\d+(?:\.\d+)?"
    for i, line in enumerate(lines):
        prev_line = lines[i - 1] if i > 0 else ""
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        ctx = " ".join([prev_line, line, next_line]).strip()
        for key, (patterns, is_float) in labels.items():
            if key in out["home"] and key in out["away"]:
                continue
            cast = float if is_float else lambda x: int(round(float(x)))
            matched = False
            for label_re in patterns:
                if not re.search(label_re, line, flags=re.I):
                    continue
                if re.fullmatch(number_re, prev_line) and re.fullmatch(
                    number_re, next_line
                ):
                    try:
                        out["home"][key] = cast(prev_line)
                        out["away"][key] = cast(next_line)
                        matched = True
                        break
                    except Exception:
                        pass
                m = re.search(
                    rf"(?P<h>{number_re}).*?(?:{label_re}).*?(?P<a>{number_re})",
                    ctx,
                    flags=re.I,
                )
                if m:
                    try:
                        out["home"][key] = cast(m.group("h"))
                        out["away"][key] = cast(m.group("a"))
                        matched = True
                        break
                    except Exception:
                        pass
            if matched:
                continue

    out = _finalize_official_stats(out)
    return out if _official_stats_score(out) > 0 else {}


def _to_num(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("%", "").replace(",", ".")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return None


def _extract_official_stats_from_dom(driver) -> dict:
    """Extract official visible left/right numbers from the rendered WhoScored page DOM."""
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception:
        return {}

    wanted = [
        "Total Team xG",
        "Shots",
        "Shots on target",
        "Shots off target",
        "Shots blocked",
        "Woodwork",
    ]

    try:
        WebDriverWait(driver, 30).until(
            lambda d: "Total Team xG" in d.find_element(By.TAG_NAME, "body").text
        )
    except Exception:
        pass

    js = r"""
    const wanted = arguments[0];

    function clean(txt){
        return (txt || "").replace(/\s+/g, " ").trim();
    }

    function isNumeric(txt){
        txt = clean(txt).replace('%','').replace(',', '.');
        return /^-?\d+(\.\d+)?$/.test(txt);
    }

    function visible(el){
        if (!el) return false;
        const st = window.getComputedStyle(el);
        if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    }

    function collectTextNodes(root){
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null);
        const out = [];
        let node = walker.currentNode;
        while(node){
            if (visible(node)){
                const txt = clean(node.innerText);
                if (txt) out.push({el: node, text: txt});
            }
            node = walker.nextNode();
        }
        return out;
    }

    const nodes = collectTextNodes(document.body);
    const results = {};

    for (const label of wanted){
        let found = null;

        for (const item of nodes){
            if (item.text === label){
                found = item.el;
                break;
            }
        }

        if (!found){
            for (const item of nodes){
                if (item.text.includes(label)){
                    found = item.el;
                    break;
                }
            }
        }

        if (!found) continue;

        let container = found.parentElement;
        let best = null;

        for (let i = 0; i < 6 && container; i++, container = container.parentElement){
            const txt = clean(container.innerText);
            if (!txt.includes(label)) continue;

            const lines = txt.split('\n').map(x => clean(x)).filter(Boolean);

            let idx = lines.findIndex(x => x === label || x.includes(label));
            if (idx !== -1){
                let before = null, after = null;

                for (let j = idx - 1; j >= 0; j--){
                    if (isNumeric(lines[j])) { before = lines[j]; break; }
                }
                for (let j = idx + 1; j < lines.length; j++){
                    if (isNumeric(lines[j])) { after = lines[j]; break; }
                }

                if (before !== null && after !== null){
                    best = {home: before, away: after, raw: txt};
                    break;
                }
            }

            const nums = lines.filter(isNumeric);
            if (nums.length >= 2){
                best = {home: nums[0], away: nums[1], raw: txt};
                break;
            }
        }

        if (best) results[label] = best;
    }

    return results;
    """

    try:
        raw = driver.execute_script(js, wanted)
    except Exception:
        return {}

    stats = {}
    for label, vals in (raw or {}).items():
        stats[label] = {
            "home": _to_num((vals or {}).get("home")),
            "away": _to_num((vals or {}).get("away")),
        }

    required = ["Total Team xG", "Shots", "Shots on target"]
    missing = [
        k
        for k in required
        if k not in stats or stats[k]["home"] is None or stats[k]["away"] is None
    ]
    if missing:
        return {}

    shots_h = stats["Shots"]["home"]
    shots_a = stats["Shots"]["away"]
    ont_h = stats["Shots on target"]["home"]
    ont_a = stats["Shots on target"]["away"]

    off_h = (stats.get("Shots off target") or {}).get("home")
    off_a = (stats.get("Shots off target") or {}).get("away")
    blk_h = (stats.get("Shots blocked") or {}).get("home")
    blk_a = (stats.get("Shots blocked") or {}).get("away")

    if off_h is None and blk_h is not None:
        off_h = shots_h - ont_h - blk_h
    if off_a is None and blk_a is not None:
        off_a = shots_a - ont_a - blk_a
    if blk_h is None and off_h is not None:
        blk_h = shots_h - ont_h - off_h
    if blk_a is None and off_a is not None:
        blk_a = shots_a - ont_a - off_a

    wood_h = (stats.get("Woodwork") or {}).get("home")
    wood_a = (stats.get("Woodwork") or {}).get("away")

    out = {
        "home": {
            "xG": float(stats["Total Team xG"]["home"]),
            "shots": int(shots_h),
            "on_target": int(ont_h),
            "off_target": int(off_h if off_h is not None else 0),
            "blocked": int(blk_h if blk_h is not None else 0),
            "woodwork": int(wood_h if wood_h is not None else 0),
        },
        "away": {
            "xG": float(stats["Total Team xG"]["away"]),
            "shots": int(shots_a),
            "on_target": int(ont_a),
            "off_target": int(off_a if off_a is not None else 0),
            "blocked": int(blk_a if blk_a is not None else 0),
            "woodwork": int(wood_a if wood_a is not None else 0),
        },
    }
    return _finalize_official_stats(out)


def _start_browser_driver(
    chromedriver_path: str = None,
    profile_dir: str = None,
    profile_name: str = "Default",
):
    """
    Start an isolated Chrome session for DOM extraction.
    Root fix for DevToolsActivePort failures:
    - do NOT reuse the user's live Chrome profile by default
    - use a fresh temporary user-data-dir
    - enable remote debugging port explicitly
    - optionally run headless for stability
    """
    temp_user_data_dir = tempfile.mkdtemp(prefix="ws_dom_profile_")

    def _common_args(opts):
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--lang=en-US")
        opts.add_argument("--remote-debugging-port=0")
        opts.add_argument("--no-first-run")
        opts.add_argument("--no-default-browser-check")
        opts.add_argument("--disable-background-networking")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--disable-popup-blocking")
        opts.add_argument("--disable-notifications")
        if BROWSER_HEADLESS:
            opts.add_argument("--headless=new")

        # Use an isolated temp profile by default.
        # Only opt into the real profile if explicitly enabled.
        if BROWSER_USE_REAL_PROFILE and profile_dir and os.path.isdir(profile_dir):
            opts.add_argument(f"--user-data-dir={profile_dir}")
            opts.add_argument(f"--profile-directory={profile_name}")
        else:
            opts.add_argument(f"--user-data-dir={temp_user_data_dir}")
            opts.add_argument("--profile-directory=Default")

    try:
        import undetected_chromedriver as uc

        _patch_undetected_chromedriver_del(uc)
        opts = uc.ChromeOptions()
        _common_args(opts)
        opts.add_argument("--disable-blink-features=AutomationControlled")
        # Force the detected Chrome major version so uc does not download a mismatched future driver.
        driver = uc.Chrome(**_uc_chrome_kwargs(opts))
        return driver, "uc", temp_user_data_dir
    except Exception as _uc_err:
        console.print(f"[dim]  uc auto-download failed: {_uc_err}[/dim]")

    if chromedriver_path and os.path.exists(chromedriver_path):
        try:
            import undetected_chromedriver as uc

            _patch_undetected_chromedriver_del(uc)
            opts = uc.ChromeOptions()
            _common_args(opts)
            opts.add_argument("--disable-blink-features=AutomationControlled")
            driver = uc.Chrome(**_uc_chrome_kwargs(opts, chromedriver_path))
            return driver, "uc-manual", temp_user_data_dir
        except Exception:
            pass

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    opts = Options()
    _common_args(opts)
    if chromedriver_path and os.path.exists(chromedriver_path):
        try:
            driver = webdriver.Chrome(service=Service(chromedriver_path), options=opts)
            return driver, "selenium-manual", temp_user_data_dir
        except Exception:
            pass
    driver = webdriver.Chrome(options=opts)
    return driver, "selenium", temp_user_data_dir


def _get_official_stats(
    info: dict,
    url: str,
    chromedriver_path: str = None,
    profile_dir: str = None,
    profile_name: str = "Default",
) -> dict:
    # Root fix: avoid Chrome entirely unless explicitly enabled.
    http_stats = _extract_official_stats_http_only(url)
    if _official_stats_has(http_stats):
        console.print(
            "[green]  Official WhoScored/Opta stats captured via HTTP/HTML only (browser skipped).[/green]"
        )
        return _finalize_official_stats(http_stats)

    if not BROWSER_DOM_FALLBACK_ENABLED:
        raise RuntimeError(
            "Official WhoScored/Opta stats were not captured via HTTP/HTML parsing. "
            "Browser fallback is disabled in this root-fix build because Chrome startup is what is failing on your machine. "
            "Define BROWSER_DOM_FALLBACK_ENABLED = True only if you want to retry Chrome manually. "
            "The script will not use fallback event totals in strict official mode."
        )

    driver = None
    temp_user_data_dir = None
    last_error = None
    try:
        driver, driver_kind, temp_user_data_dir = _start_browser_driver(
            chromedriver_path, profile_dir, profile_name
        )
        console.print(
            f"[cyan]  Capturing official WhoScored/Opta stats via {driver_kind} DOM...[/cyan]"
        )
        driver.get("https://www.whoscored.com/")
        time.sleep(2)
        driver.get(url)

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait

            WebDriverWait(driver, 90).until(
                lambda d: "matchCentreData" in d.page_source
            )

            phrases = ["statistics", "summary", "stats"]
            xpath_tpl = (
                "//*[self::a or self::button or self::li or self::span or self::div]"
                "[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{p}')]"
            )
            for _ in range(18):
                try:
                    body_text = driver.find_element(By.TAG_NAME, "body").text or ""
                except Exception:
                    body_text = ""
                if "Total Team xG" in body_text and "Shots on target" in body_text:
                    break
                for phrase in phrases:
                    try:
                        elems = driver.find_elements(
                            By.XPATH, xpath_tpl.format(p=phrase)
                        )
                    except Exception:
                        elems = []
                    for el in elems[:12]:
                        try:
                            driver.execute_script(
                                "arguments[0].scrollIntoView({block:'center'});", el
                            )
                            time.sleep(0.1)
                            driver.execute_script("arguments[0].click();", el)
                        except Exception:
                            pass
                time.sleep(0.7)

            WebDriverWait(driver, 30).until(
                lambda d: "Total Team xG" in d.find_element(By.TAG_NAME, "body").text
            )
        except Exception as e:
            last_error = e

        try:
            visible_text = (
                driver.execute_script(
                    "return document.body ? document.body.innerText : ''; "
                )
                or ""
            )
        except Exception:
            visible_text = ""
        try:
            html = driver.page_source
        except Exception:
            html = ""
        _capture_scraped_page(html, visible_text)

        dom_stats = _extract_official_stats_from_dom(driver)
        if _official_stats_has(dom_stats):
            return _finalize_official_stats(dom_stats)

        txt_stats = _extract_official_stats_from_text(visible_text)
        if _official_stats_has(txt_stats):
            return _finalize_official_stats(txt_stats)

        html_stats = _merge_official_stats(
            _extract_official_stats_from_initialdata(html),
            _extract_official_stats_from_text(html),
        )
        if _official_stats_has(html_stats):
            return _finalize_official_stats(html_stats)

    except Exception as e:
        last_error = e
    finally:
        _safe_quit_driver(driver)
        try:
            if temp_user_data_dir and os.path.isdir(temp_user_data_dir):
                shutil.rmtree(temp_user_data_dir, ignore_errors=True)
        except Exception:
            pass

    extra = f" Last browser error: {last_error}" if last_error else ""
    raise RuntimeError(
        "Official WhoScored/Opta stats were not captured from HTTP/HTML parsing or the rendered page DOM."
        " The script is running in strict official mode and will not use fallback event totals."
        + extra
    )


def _event_stat_count(events: pd.DataFrame, team_id: int, stat: str) -> int:
    if events is None or events.empty or team_id is None:
        return 0
    shots = events[
        (events.get("is_shot") == True) & (events.get("team_id") == team_id)
    ].copy()
    if "is_own_goal" in shots.columns:
        shots = shots[~shots["is_own_goal"].fillna(False)]
    if "is_penalty_shootout" in shots.columns:
        shots = shots[~shots["is_penalty_shootout"].fillna(False)]
    if stat == "shots":
        return int(len(shots))
    if stat == "big_chances":
        return (
            int(shots.get("big_chance", pd.Series(dtype=bool)).fillna(False).sum())
            if "big_chance" in shots.columns
            else 0
        )
    if stat == "penalties":
        return (
            int(shots.get("is_penalty", pd.Series(dtype=bool)).fillna(False).sum())
            if "is_penalty" in shots.columns
            else 0
        )
    if stat == "headers":
        return (
            int(shots.get("is_header", pd.Series(dtype=bool)).fillna(False).sum())
            if "is_header" in shots.columns
            else 0
        )
    if stat == "on_target":
        col = shots.get(
            "shot_whoscored_type", shots.get("shot_category", pd.Series(dtype=str))
        )
        return int(col.isin(["Goal", "SavedShot", "On Target"]).sum())
    if stat == "woodwork":
        col = shots.get("shot_whoscored_type", pd.Series(dtype=str))
        return int(col.isin(["ShotOnPost"]).sum())
    return 0


def _side_event_shots(events: pd.DataFrame, team_id: int) -> pd.DataFrame:
    if events is None or events.empty or team_id is None:
        return pd.DataFrame()
    s = events[
        (events.get("is_shot") == True) & (events.get("team_id") == team_id)
    ].copy()
    if "is_own_goal" in s.columns:
        s = s[~s["is_own_goal"].fillna(False)]
    if "is_penalty_shootout" in s.columns:
        s = s[~s["is_penalty_shootout"].fillna(False)]
    return s


def _pick_stat_value(info: dict, side: str, events: pd.DataFrame, stat: str):
    official_side = (info.get("official_stats", {}) or {}).get(side, {}) or {}
    matchcentre_side = (info.get("matchcentre_stats", {}) or {}).get(side, {}) or {}
    for src in (official_side, matchcentre_side):
        val = src.get(stat)
        if val is not None:
            try:
                return int(round(float(val)))
            except Exception:
                pass
    tid = info.get(f"{side}_id")
    return _event_stat_count(events, tid, stat)


def _estimate_public_site_xg_total_for_side(
    info: dict, events: pd.DataFrame, side: str
) -> float | None:
    """Estimate an internal team-stat xG target when no external xG is allowed."""
    tid = info.get(f"{side}_id")
    shots_df = _side_event_shots(events, tid)
    n = int(len(shots_df))
    if n == 0:
        return None

    raw_total = float(
        pd.to_numeric(shots_df.get("xG", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .sum()
    )
    if raw_total <= 0:
        raw_total = float(
            sum(_opta_like_local_xg_from_row(r) for _, r in shots_df.iterrows())
        )

    shots = _pick_stat_value(info, side, events, "shots") or n
    big = _pick_stat_value(info, side, events, "big_chances") or 0
    on_target = _pick_stat_value(info, side, events, "on_target") or _event_stat_count(
        events, tid, "on_target"
    )
    woodwork = _pick_stat_value(info, side, events, "woodwork") or _event_stat_count(
        events, tid, "woodwork"
    )
    penalties = _event_stat_count(events, tid, "penalties")

    box_shots = 0
    central_box_shots = 0
    six_yard_shots = 0
    long_shots = 0
    headers = 0
    crosses = 0
    direct_fks = 0
    rebounds = 0
    cutbacks = 0
    through_balls = 0

    for _, r in shots_df.iterrows():
        f = _shot_geometry_features(r)
        ctx = _shot_context_features(r, f)
        box_shots += int(bool(f["in_box"]))
        central_box_shots += int(bool(f["central_box"]))
        six_yard_shots += int(bool(f["in_six"]))
        long_shots += int(f["distance"] > 20.0)
        headers += int(ctx["is_header"])
        crosses += int(ctx["is_cross"])
        direct_fks += int(ctx["is_direct_fk"])
        rebounds += int(ctx["is_rebound"])
        cutbacks += int(ctx["is_layoff"])
        through_balls += int(ctx["is_through"])

    non_penalty_shots = max(shots - penalties, 0)

    # Team-stat prior built only from available match/event data. It intentionally
    # does not use goals scored, so finishing outcome does not contaminate chance quality.
    prior = (
        0.026 * non_penalty_shots
        + 0.026 * box_shots
        + 0.052 * central_box_shots
        + 0.105 * six_yard_shots
        + XG_INTERNAL_BIG_CHANCE_VALUE * big
        + XG_INTERNAL_ON_TARGET_BONUS * on_target
        + XG_INTERNAL_WOODWORK_BONUS * woodwork
        + 0.730 * penalties
        + 0.048 * rebounds
        + 0.052 * cutbacks
        + 0.039 * through_balls
    )

    # Penalise profiles that are noisy in xG models: many headers/crosses, direct free kicks,
    # and long-shot-heavy shot maps.
    if headers >= max(2, shots * 0.34):
        prior *= 0.88
    if crosses >= max(2, shots * 0.40):
        prior *= 0.91
    if long_shots >= max(3, shots * 0.45):
        prior *= 0.84
    if direct_fks:
        prior -= min(0.040 * direct_fks, 0.14)

    # Big chances should set a floor, but not explode the total.
    if big:
        big_floor = big * 0.250 + max(shots - big, 0) * 0.022 + penalties * 0.28
        prior = max(prior, big_floor)

    w = float(_clamp(XG_INTERNAL_TEAM_PRIOR_WEIGHT, 0.0, 0.85))
    target = (1.0 - w) * raw_total + w * prior
    lower = raw_total * XG_INTERNAL_TARGET_MULTIPLIER_MIN
    upper = raw_total * XG_INTERNAL_TARGET_MULTIPLIER_MAX
    if big:
        upper = max(upper, prior * 1.08)
    target = _clamp(target, max(0.01, lower), max(0.01, upper))
    target = min(target, n * XG_SINGLE_SHOT_CAP)
    return round(float(target), 2)


def _build_public_site_fallback_stats(info: dict, events: pd.DataFrame) -> dict:
    if not XG_USE_INTERNAL_TEAM_STAT_CALIBRATION:
        return {}
    out = {"home": {}, "away": {}}
    for side in ("home", "away"):
        xg = _estimate_public_site_xg_total_for_side(info, events, side)
        if xg is not None:
            out[side]["xG"] = xg
            out[side]["shots"] = _pick_stat_value(info, side, events, "shots")
            out[side]["on_target"] = _pick_stat_value(info, side, events, "on_target")
            out[side]["big_chances"] = _pick_stat_value(
                info, side, events, "big_chances"
            )
    return _finalize_official_stats(out)


def _fill_missing_xg_with_public_fallback(info: dict, events: pd.DataFrame) -> dict:
    """Fill xG totals from the V7 internal team-stat target, not from external totals."""
    current = _finalize_official_stats(info.get("official_stats", {}) or {})
    fallback = _build_public_site_fallback_stats(info, events)
    used = False
    for side in ("home", "away"):
        cur_side = current.setdefault(side, {})
        fb_side = fallback.get(side, {}) or {}
        if cur_side.get("xG") is None and fb_side.get("xG") is not None:
            cur_side["xG"] = fb_side.get("xG")
            used = True
        for k in ("shots", "on_target", "big_chances"):
            if cur_side.get(k) is None and fb_side.get(k) is not None:
                cur_side[k] = fb_side.get(k)
    if used:
        info["xg_reference_source"] = "v7 internal event/team-stat model"
        try:
            console.print(
                "[yellow]  Using V7 internal xG team-stat calibration; no external xG total is used.[/yellow]"
            )
        except Exception:
            pass
    return _finalize_official_stats(current)


def _bounded_rescale_to_total(
    values, target_total: float, cap: float = XG_SINGLE_SHOT_CAP
) -> list[float]:
    """Scale shot xG values to an official team total while keeping single-shot bounds."""
    vals = [float(_normalise_xg_value(v) or 0.001) for v in values]
    n = len(vals)
    if n == 0:
        return []
    target_total = float(max(0.0, target_total))
    max_possible = cap * n
    target_total = min(target_total, max_possible)

    if sum(vals) <= 0:
        vals = [target_total / n] * n
    else:
        factor = target_total / sum(vals)
        vals = [v * factor for v in vals]

    fixed = [False] * n
    for _ in range(12):
        changed = False
        for i, v in enumerate(vals):
            if vals[i] > cap:
                vals[i] = cap
                fixed[i] = True
                changed = True
            elif vals[i] < 0.001:
                vals[i] = 0.001
                fixed[i] = True
                changed = True
        remaining = [i for i in range(n) if not fixed[i]]
        if not remaining:
            break
        diff = target_total - sum(vals)
        if abs(diff) < 1e-10:
            break
        base = sum(vals[i] for i in remaining)
        if base <= 0:
            add = diff / len(remaining)
            for i in remaining:
                vals[i] += add
        else:
            for i in remaining:
                vals[i] += diff * (vals[i] / base)
        if not changed and abs(target_total - sum(vals)) < 1e-8:
            break

    vals = [round(float(_clamp(v, 0.001, cap)), 4) for v in vals]
    drift = round(target_total - sum(vals), 4)
    if vals:
        # Put tiny rounding drift on the largest non-capped shot when possible.
        order = sorted(range(n), key=lambda i: vals[i], reverse=True)
        for i in order:
            candidate = round(vals[i] + drift, 4)
            if 0.001 <= candidate <= cap:
                vals[i] = candidate
                break
    return vals


def _apply_official_stats_calibration(info: dict, events: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale each team's shot values to the chosen team xG target.

    In V7 the target is internal: it is produced from event/match statistics only.
    Official/provider totals are ignored unless XG_USE_OFFICIAL_TEAM_TOTAL_CALIBRATION
    is deliberately turned back on.
    """
    if (
        events is None
        or events.empty
        or not (
            XG_USE_OFFICIAL_TEAM_TOTAL_CALIBRATION
            or XG_USE_INTERNAL_TEAM_STAT_CALIBRATION
        )
    ):
        return events

    official = info.get("official_stats", {}) or {}
    out = events.copy()
    if "xg_source" not in out.columns:
        out["xg_source"] = ""

    for side in ("home", "away"):
        tid = info.get(f"{side}_id")
        target_xg = (official.get(side, {}) or {}).get("xG")
        if tid is None or target_xg is None:
            continue

        mask = (out["is_shot"] == True) & (out["team_id"] == tid)
        if "is_own_goal" in out.columns:
            mask &= ~out["is_own_goal"].fillna(False)
        if "is_penalty_shootout" in out.columns:
            mask &= ~out["is_penalty_shootout"].fillna(False)
        idx = list(out.index[mask])
        if not idx:
            continue

        scaled = _bounded_rescale_to_total(
            out.loc[idx, "xG"].fillna(0.001).tolist(), float(target_xg)
        )
        out.loc[idx, "xG"] = scaled
        out.loc[idx, "xg_source"] = out.loc[idx, "xg_source"].astype(str).replace(
            "", XG_LOCAL_MODEL_VERSION
        ) + (
            "__team_total_calibrated_to_internal_v7"
            if XG_USE_INTERNAL_TEAM_STAT_CALIBRATION
            and not XG_USE_OFFICIAL_TEAM_TOTAL_CALIBRATION
            else "__team_total_calibrated_to_official_opta"
        )

    return out


def _build_player_meta(players, events, sub_in, sub_out):
    """Per-player metadata (position, shirt, is_sub) keyed by player_id.

    `is_sub` (was this player a substitute, i.e. came off the bench) is read
    from match flow, not a single flag, so it stays correct even for 1st-half
    substitutions and feeds with an unreliable isFirstEleven field. Priority:
        1. Explicit sub events — a player who came ON is a sub; a starter who
           was subbed OFF is NOT a sub (he started).
        2. A sane isFirstEleven lineup (7–11 flagged starters) → trust it.
        3. Fallback: the 11 players with the earliest first-touch minute are
           the starters; everyone else came on later → substitute.
    """
    sub_in = set(sub_in or [])
    sub_out = set(sub_out or [])
    try:
        first_min = events.groupby("player_id")["minute"].min().to_dict()
    except Exception:
        first_min = {}
    meta = {}
    team_ids = {p.get("team_id") for p in players}
    for tid in team_ids:
        tps = [
            p
            for p in players
            if p.get("team_id") == tid and p.get("player_id") is not None
        ]
        flagged = [p for p in tps if p.get("is_first_xi")]
        use_flag = 7 <= len(flagged) <= 11
        starter_ids = set()
        if not use_flag and tps:
            ranked = sorted(tps, key=lambda p: first_min.get(p["player_id"], 9999))
            starter_ids = {p["player_id"] for p in ranked[:11]}
        for p in tps:
            pid = p["player_id"]
            if pid in sub_in:
                is_sub = True
            elif pid in sub_out:
                is_sub = False
            elif use_flag:
                is_sub = not bool(p.get("is_first_xi"))
            else:
                is_sub = pid not in starter_ids
            meta[pid] = {
                "position": p.get("position"),
                "shirt": p.get("shirt_no"),
                "is_sub": is_sub,
            }
    return meta


def _extract_match_date(md: dict) -> str:
    """The KICK-OFF date of the match (not the report date), pulled from
    WhoScored's matchCentreData. Tries the several keys the feed uses and
    returns a clean YYYY-MM-DD string, or '' if none is present."""
    import re as _re

    for key in (
        "startDate",
        "startTime",
        "matchStartDate",
        "startDateStr",
        "kickoffTime",
        "date",
    ):
        raw = md.get(key)
        if not raw:
            continue
        s = str(raw).strip()
        # ISO-ish "2026-06-22T18:00:00" / "2026-06-22 18:00:00"
        m = _re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # "22/06/2026" or "22-06-2026"
        m = _re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
        if m:
            return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return ""


def parse_all(md: dict):

    home = md.get("home", {})
    away = md.get("away", {})
    info = {
        "home_name": home.get("name"),
        "away_name": away.get("name"),
        "home_id": home.get("teamId"),
        "away_id": away.get("teamId"),
        "score": md.get("score", "? - ?"),
        "venue": md.get("venueName", ""),
        "date": _extract_match_date(md),
        "home_form": (home.get("formations") or [{}])[0].get("formationName", "N/A"),
        "away_form": (away.get("formations") or [{}])[0].get("formationName", "N/A"),
        "matchcentre_stats": _extract_matchcentre_stats(md),
    }
    pnames = {int(k): v for k, v in md.get("playerIdNameDictionary", {}).items()}
    rows = []
    sub_in, sub_out, red_cards = set(), set(), set()

    for e in md.get("events", []):
        quals = e.get("qualifiers", [])

        def dn(field):
            v = e.get(field, {})
            return v.get("displayName") if isinstance(v, dict) else v

        etype = dn("type")
        period_raw = dn("period") or ""
        period_code = PERIOD_CODES.get(period_raw, period_raw.lower())
        qual_names = [q.get("type", {}).get("displayName", "") for q in quals]
        is_penalty_shootout = _is_penalty_shootout_period(period_raw, period_code)
        if not is_penalty_shootout:
            joined_quals = " ".join(str(q) for q in qual_names).lower().replace(" ", "")
            is_penalty_shootout = (
                "penaltyshootout" in joined_quals or "shootout" in joined_quals
            )

        is_shot = (
            e.get("isShot", False) or (etype in SHOT_TYPES)
        ) and not is_penalty_shootout
        is_pass = etype in ["Pass", "OffsidPass", "KeyPass"]

        if is_shot and has_q(quals, "Blocked"):
            shot_raw_type = "BlockedShot"
        else:
            shot_raw_type = etype if is_shot and etype in SHOT_TYPES else None
        shot_cat = get_shot_family(shot_raw_type) if is_shot else None
        xg_val = _extract_provider_shot_xg(e) if is_shot else None
        assist_id = next(
            (
                int(q["value"])
                for q in quals
                if q.get("type", {}).get("displayName") == "IntentionalAssist"
                and q.get("value") is not None
            ),
            None,
        )
        pid = e.get("playerId")
        event_team = e.get("teamId")

        if etype == "SubstitutionOn" and pid:
            sub_in.add(pid)
        if etype == "SubstitutionOff" and pid:
            sub_out.add(pid)
        if etype == "Card" and pid:
            for q in quals:
                if q.get("type", {}).get("displayName") == "Red":
                    red_cards.add(pid)

        is_own_goal = (etype == "OwnGoal") or ("OwnGoal" in qual_names)
        is_goal_flag = (
            e.get("isGoal", False) or is_own_goal
        ) and not is_penalty_shootout
        scoring_team = (
            (info["away_id"] if event_team == info["home_id"] else info["home_id"])
            if is_own_goal
            else event_team
        )

        is_cross = is_pass and has_q(quals, "Cross")

        rows.append(
            {
                "event_id": e.get("id"),
                "period": period_raw,
                "period_code": period_code,
                "minute": e.get("minute"),
                "second": e.get("second", 0),
                "team_id": event_team,
                "player_id": pid,
                "player": pnames.get(pid, ""),
                "type": etype,
                "outcome": dn("outcomeType"),
                "x": e.get("x"),
                "y": e.get("y"),
                "end_x": e.get("endX"),
                "end_y": e.get("endY"),
                "is_shot": is_shot,
                "is_pass": is_pass,
                "is_key_pass": has_q(quals, "KeyPass"),
                "is_cross": is_cross,
                "shot_category": shot_cat,
                "shot_whoscored_type": shot_raw_type,
                "is_goal": is_goal_flag,
                "is_own_goal": is_own_goal,
                "scoring_team": scoring_team,
                "is_header": has_q(quals, "Head"),
                "is_penalty": has_q(quals, "Penalty"),
                "is_penalty_shootout": is_penalty_shootout,
                "big_chance": has_q(quals, "BigChance"),
                "body_part": next(
                    (
                        q.get("type", {}).get("displayName")
                        for q in quals
                        if q.get("type", {}).get("displayName")
                        in ["Head", "RightFoot", "LeftFoot"]
                    ),
                    None,
                ),
                "assist_player": pnames.get(assist_id, "") if assist_id else "",
                "qualifier_names": qual_names,
                "is_direct_fk": _is_direct_freekick_row(
                    {
                        "qualifier_names": qual_names,
                        "type": etype,
                        "is_direct_fk": False,
                    }
                ),
                "assist_type": next(
                    (
                        q.get("type", {}).get("displayName")
                        for q in quals
                        if q.get("type", {}).get("displayName")
                        in ["KeyPass", "ThroughBall", "Cross", "Chipped", "FastBreak"]
                    ),
                    None,
                ),
                "xG": xg_val,
                "xg_source": "provider_shot_xg" if xg_val is not None else "",
                "xT": (
                    calc_xt_pass(e.get("x"), e.get("y"), e.get("endX"), e.get("endY"))
                    if is_pass
                    else None
                ),
            }
        )

    events = pd.DataFrame(rows)
    if not events.empty:

        def _event_role(pid):
            if pid in red_cards:
                return "red_card"
            if pid in sub_in and pid in sub_out:
                return "both_sub"
            if pid in sub_in:
                return "sub_in"
            if pid in sub_out:
                return "sub_out"
            return ""

        events["player_role"] = events["player_id"].map(_event_role)
        events = apply_best_open_source_xg(events, info)
    players = []
    for side in ["home", "away"]:
        t = md.get(side, {})
        for p in t.get("players", []):
            stats = p.get("stats", {})
            players.append(
                {
                    "player_id": p.get("playerId"),
                    "name": p.get("name"),
                    "position": p.get("position"),
                    "shirt_no": p.get("shirtNo"),
                    "team_name": t.get("name"),
                    "team_id": t.get("teamId"),
                    "side": side,
                    "is_first_xi": p.get("isFirstEleven", False),
                    "rating": p.get("playerScore"),
                    "touches": (stats.get("touches") or {}).get("total"),
                    "passes": (stats.get("passesTotal") or {}).get("total"),
                }
            )

    info["sub_in"] = sub_in
    info["sub_out"] = sub_out
    info["red_cards"] = red_cards
    # Per-player metadata (real position + shirt number) keyed by player_id.
    # The pass-network renderer uses this to colour nodes by actual unit
    # (not geometric depth) and to print real shirt numbers inside nodes.
    info["player_meta"] = _build_player_meta(players, events, sub_in, sub_out)
    return info, events, pd.DataFrame(players)


# ══════════════════════════════════════════════════════
#  xG STATS
# ══════════════════════════════════════════════════════
def xg_stats(events: pd.DataFrame, info: dict) -> dict:
    """
    Compute per-team xG statistics.
    Own goals are excluded from the shooting team's xG
    (they count for the *conceding* team's ledger, not the attacker's model).

    Shot buckets follow WhoScored's definitions:
      On Target = Goal + SavedShot
      Off Target = MissedShots + ShotOnPost
      Blocked = BlockedShot
    """
    shots_all = events[events["is_shot"] == True].copy()
    if "is_penalty_shootout" in shots_all.columns:
        shots_all = shots_all[~shots_all["is_penalty_shootout"].fillna(False)]
    out = {}

    for side in ["home", "away"]:
        tid = info[f"{side}_id"]
        name = info[f"{side}_name"]

        s = (
            shots_all[
                (shots_all["team_id"] == tid) & (shots_all["is_own_goal"] == False)
            ].copy()
            if "is_own_goal" in shots_all.columns
            else shots_all[shots_all["team_id"] == tid].copy()
        )

        goals_scored = (
            int(
                events[
                    (events["is_goal"] == True)
                    & (events["scoring_team"] == tid)
                    & (
                        ~events.get(
                            "is_penalty_shootout", pd.Series(False, index=events.index)
                        ).fillna(False)
                    )
                ].shape[0]
            )
            if "scoring_team" in events.columns
            else int(
                (s.get("shot_whoscored_type", s["shot_category"]).eq("Goal")).sum()
            )
        )

        raw = (
            s["shot_whoscored_type"]
            if "shot_whoscored_type" in s.columns
            else s["shot_category"]
        )
        on_target_mask = raw.isin(["Goal", "SavedShot", "On Target"])
        counts = get_shot_counts(s)

        matchcentre_side = info.get("matchcentre_stats", {}).get(side, {}) or {}
        official_side = info.get("official_stats", {}).get(side, {}) or {}

        xg_total = round(float(s["xG"].fillna(0.0).sum()), 2)
        if XG_USE_OFFICIAL_TEAM_TOTAL_CALIBRATION:
            if matchcentre_side.get("xG") is not None:
                xg_total = round(float(matchcentre_side["xG"]), 2)
            if official_side.get("xG") is not None:
                xg_total = round(float(official_side["xG"]), 2)
        elif (
            XG_USE_INTERNAL_TEAM_STAT_CALIBRATION
            and official_side.get("xG") is not None
        ):
            # This xG is produced by the internal V7 team-stat model, not by an external site.
            xg_total = round(float(official_side["xG"]), 2)

        for source_side in (matchcentre_side, official_side):
            if source_side.get("shots") is not None:
                counts["shots"] = int(source_side["shots"])
            if source_side.get("on_target") is not None:
                src_on_target = int(source_side["on_target"])
                counts["on_target"] = src_on_target
                counts["saved"] = max(src_on_target - goals_scored, 0)
            if source_side.get("off_target") is not None:
                counts["off_target"] = int(source_side["off_target"])
                counts["missed"] = max(
                    int(source_side["off_target"])
                    - int(source_side.get("woodwork", counts["post"])),
                    0,
                )
            if source_side.get("blocked") is not None:
                counts["blocked"] = int(source_side["blocked"])
            if source_side.get("woodwork") is not None:
                counts["post"] = int(source_side["woodwork"])

        xgot = round(float(s[on_target_mask]["xG"].fillna(0).sum()), 2)
        raw_on_target = int(on_target_mask.sum())
        if counts["on_target"] != raw_on_target and raw_on_target > 0:
            xgot = round(xgot * (counts["on_target"] / raw_on_target), 2)
        xgot = min(xgot, xg_total)
        xg_per_shot = round(xg_total / max(counts["shots"], 1), 3)

        team_passes = (
            events[
                (events["is_pass"] == True)
                & (events["team_id"] == tid)
                & (events["outcome"] == "Successful")
                & events["xT"].notna()
            ]
            if "xT" in events.columns
            else pd.DataFrame()
        )
        xt_total = (
            round(float(team_passes["xT"].sum()), 3) if not team_passes.empty else 0.0
        )

        out[name] = {
            "xG": xg_total,
            "xGoT": xgot,
            "xG_per_shot": xg_per_shot,
            "shots": counts["shots"],
            "on_target": counts["on_target"],
            "goals": goals_scored,
            "saved": counts["saved"],
            "missed": counts["missed"],
            "off_target": counts["off_target"],
            "blocked": counts["blocked"],
            "post": counts["post"],
            "big_chances": (
                int(s["big_chance"].sum()) if "big_chance" in s.columns else 0
            ),
            "xT": xt_total,
        }
    return out


# ══════════════════════════════════════════════════════
#  PASS NETWORK BUILDER
# ══════════════════════════════════════════════════════
def build_pass_network(events: pd.DataFrame, team_id):
    team_evts = (
        events[events["team_id"] == team_id]
        .sort_values(["minute", "second"])
        .reset_index(drop=True)
    )
    nodes, edges = {}, {}
    passes = team_evts[team_evts["is_pass"] == True]
    for pid, grp in passes.groupby("player_id"):
        nodes[pid] = {
            "name": grp["player"].iloc[0],
            "avg_x": grp["x"].mean(),
            "avg_y": grp["y"].mean(),
            "pass_count": len(grp),
        }
    succ = team_evts[
        (team_evts["is_pass"] == True) & (team_evts["outcome"] == "Successful")
    ].copy()
    for i in range(len(succ)):
        curr_idx = succ.index[i]
        passer_id = succ.iloc[i]["player_id"]
        later = team_evts[(team_evts.index > curr_idx) & team_evts["player_id"].notna()]
        if later.empty:
            continue
        recv_id = later.iloc[0]["player_id"]
        if passer_id == recv_id:
            continue
        if recv_id not in nodes:
            rr = team_evts[team_evts["player_id"] == recv_id]
            if not rr.empty:
                nodes[recv_id] = {
                    "name": rr["player"].iloc[0],
                    "avg_x": rr["x"].mean(),
                    "avg_y": rr["y"].mean(),
                    "pass_count": 0,
                }
        key = tuple(sorted([passer_id, recv_id]))
        edges[key] = edges.get(key, 0) + 1
    return nodes, edges


def _player_role_color(pid, team_color, sub_in, sub_out, red_cards):
    if pid in red_cards:
        return COLOR_RED_CARD
    if pid in sub_in and pid in sub_out:
        return COLOR_BOTH_SUB
    if pid in sub_in:
        return COLOR_SUB_IN
    if pid in sub_out:
        return COLOR_SUB_OUT
    return team_color


def _player_role_badge(pid, sub_in, sub_out, red_cards):
    if pid in red_cards:
        return "🟥"
    if pid in sub_in and pid in sub_out:
        return "↕"
    if pid in sub_in:
        return "↑"
    if pid in sub_out:
        return "↓"
    return ""


# ══════════════════════════════════════════════════════
#  PITCH
# ══════════════════════════════════════════════════════
def draw_pitch(ax, pitch_color=None, line_color=None, line_alpha=0.82):
    pc = pitch_color or PITCH_COL
    ax.set_facecolor(pc)
    is_light = pc in ("white", "#ffffff", "#f5f5f5", "#fafafa")
    lc = line_color or ("black" if is_light else "#3d8a3d")
    lw = 1.15

    def L(*args, **kw):
        a = kw.pop("alpha", line_alpha)
        ax.plot(*args, color=lc, linewidth=lw, alpha=a, **kw)

    ax.plot(
        [0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, lw=1.8, alpha=line_alpha
    )
    L([50, 50], [0, 100], linestyle="--", alpha=0.45)
    ax.add_patch(
        plt.Circle((50, 50), 9.15 / 0.68, color=lc, fill=False, lw=lw, alpha=0.45)
    )
    ax.plot(50, 50, "o", color=lc, ms=2.5, alpha=line_alpha)
    L([0, 16.5, 16.5, 0], [21.1, 21.1, 78.9, 78.9])
    L([100, 83.5, 83.5, 100], [21.1, 21.1, 78.9, 78.9])
    L([0, 5.5, 5.5, 0], [36.8, 36.8, 63.2, 63.2])
    L([100, 94.5, 94.5, 100], [36.8, 36.8, 63.2, 63.2])
    ax.plot([11, 89], [50, 50], "o", color=lc, ms=2.5, alpha=line_alpha)
    for cx in [11, 89]:
        ax.add_patch(
            matplotlib.patches.Arc(
                (cx, 50),
                18,
                18 * (105 / 68),
                angle=0,
                theta1=-65,
                theta2=65,
                color=lc,
                lw=lw,
                alpha=0.42,
            )
        )
    gc = "black" if is_light else "white"
    ax.plot([0, 0], [44, 56], color=gc, lw=4, alpha=0.95, solid_capstyle="round")
    ax.plot([100, 100], [44, 56], color=gc, lw=4, alpha=0.95, solid_capstyle="round")
    ax.plot([0, -1.2, -1.2, 0], [44, 44, 56, 56], color=gc, lw=1.2, alpha=0.55)
    ax.plot([100, 101.2, 101.2, 100], [44, 44, 56, 56], color=gc, lw=1.2, alpha=0.55)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.axis("off")


# ══════════════════════════════════════════════════════
#  PASS ZONE HELPERS
# ══════════════════════════════════════════════════════
def _pass_zone(end_x, end_y):
    in_pen = (
        end_x is not None
        and end_y is not None
        and end_x >= PENALTY_BOX_X
        and PENALTY_BOX_Y1 <= end_y <= PENALTY_BOX_Y2
    )
    in_fin = end_x is not None and end_x >= FINAL_THIRD_X
    if in_pen:
        return "penalty"
    if in_fin:
        return "final_third"
    return "other"


def _pass_color(zone, successful):
    tbl = {
        ("penalty", True): (C_GREEN, 0.95, 5),
        ("penalty", False): (C_GOLD, 0.86, 4),
        ("final_third", True): (C_BLUE, 0.84, 3),
        ("final_third", False): (C_RED, 0.78, 2),
        ("other", True): ("#cbd5e1", 0.38, 1),
        ("other", False): ("#94a3b8", 0.30, 1),
    }
    return tbl.get((zone, successful), ("#888888", 0.2, 1))


# ══════════════════════════════════════════════════════
#  FIG 1 — xG FLOW
# ══════════════════════════════════════════════════════
def draw_xg_flow(fig, ax, events, info, xg_data, status):
    ax.clear()
    ax.set_facecolor(BG_DARK)
    shots_df = events[events["is_shot"] == True].sort_values("minute")
    live_min = int(events["minute"].dropna().max()) if not events.empty else 90
    s_label, s_color = STATUS_BADGE.get(status, ("■ Full Time", "#64748b"))
    xmax = max(live_min + 5, 97)
    if status in ("et1", "etht", "et2"):
        xmax = max(xmax, 112)
    if status in ("et2", "pso"):
        xmax = max(xmax, 130)
    if status == "pso":
        xmax = max(xmax, 148)

    periods_seen = (
        set(shots_df["period_code"].dropna().unique()) if not shots_df.empty else set()
    )
    periods_seen.update(["1h", "2h"])

    for s0, se, code_, label, zone_color, _ in PERIOD_SPANS:
        if code_ not in periods_seen:
            continue
        ax.axvspan(s0, min(se, xmax), facecolor=zone_color, alpha=0.55, zorder=0)
        ax.text(
            (s0 + min(se, xmax)) / 2,
            0,
            label,
            transform=ax.get_xaxis_transform(),
            color="#6b7280",
            fontsize=8,
            ha="center",
            va="bottom",
            fontweight="bold",
            alpha=0.7,
        )

    for pcode, base_min in STOPPAGE_PERIODS.items():
        p_shots = (
            shots_df[shots_df["period_code"] == pcode]
            if "period_code" in shots_df.columns
            else pd.DataFrame()
        )
        if p_shots.empty:
            continue
        pmax = int(p_shots["minute"].max())
        if pmax > base_min:
            ax.axvspan(
                base_min,
                pmax + 0.5,
                facecolor="#ffffff",
                alpha=0.035,
                hatch="///",
                edgecolor="#ffffff22",
                zorder=1,
            )
            ax.text(
                base_min + 0.5,
                0,
                f"+{pmax-base_min}",
                transform=ax.get_xaxis_transform(),
                color="#9ca3af",
                fontsize=7.5,
                va="bottom",
                fontweight="bold",
            )

    dividers = [(45, "HT", C_GOLD)]
    if live_min > 88 or status not in ("pre", "1h"):
        dividers.append((90, "FT", "#64748b"))
    if "et1" in periods_seen or "et2" in periods_seen:
        dividers += [(105, "ET HT", "#a855f7"), (120, "AET", "#64748b")]
    if "pso" in periods_seen:
        dividers.append((120, "PSO", C_RED))

    for xpos, label, col in dividers:
        if xpos > xmax:
            continue
        ax.axvline(xpos, color=col, linestyle="--", lw=1.5, alpha=0.75, zorder=3)
        ax.text(
            xpos + 0.5,
            0.015,
            label,
            transform=ax.get_xaxis_transform(),
            color=col,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=BG_DARK,
                alpha=0.75,
                edgecolor="none",
            ),
            zorder=5,
        )

    max_xg = 0.0
    team_color_map = {info["home_id"]: C_RED, info["away_id"]: C_BLUE}
    cumxg_by_team = {}

    for tid, name, color in [
        (info["home_id"], info["home_name"], C_RED),
        (info["away_id"], info["away_name"], C_BLUE),
    ]:
        s = shots_df[shots_df["team_id"] == tid].sort_values("minute")
        mins = [0] + s["minute"].tolist()
        cumxg = [0] + s["xG"].fillna(0).cumsum().tolist()
        max_xg = max(max_xg, cumxg[-1] if cumxg else 0)
        ax.step(mins, cumxg, where="post", color=color, lw=8, alpha=0.12, zorder=4)
        ax.step(
            mins,
            cumxg,
            where="post",
            color=color,
            lw=2.8,
            alpha=1.0,
            label=f"{name}   xG = {cumxg[-1]:.2f}",
            zorder=5,
        )
        ax.fill_between(mins, cumxg, step="post", color=color, alpha=0.10, zorder=2)
        non_goals = s[s["is_goal"] == False]
        if not non_goals.empty:
            ax.scatter(
                non_goals["minute"],
                [0] * len(non_goals),
                c=color,
                s=38,
                marker="|",
                alpha=0.60,
                zorder=6,
                clip_on=False,
            )
        cumxg_by_team[tid] = {
            "minutes": s["minute"].tolist(),
            "cumxg": s["xG"].fillna(0).cumsum().tolist(),
        }

    all_goals = events[events["is_goal"] == True].copy().sort_values("minute")
    if not all_goals.empty:
        ax.scatter(
            all_goals["minute"],
            [0] * len(all_goals),
            c=C_GOLD,
            s=110,
            marker="*",
            alpha=0.98,
            zorder=7,
            clip_on=False,
        )

    for _, row in all_goals.iterrows():
        is_og = bool(row.get("is_own_goal", False))
        beneficiary_id = row.get("scoring_team", row["team_id"])
        ann_color = team_color_map.get(beneficiary_id, C_RED)
        td = cumxg_by_team.get(beneficiary_id, {"minutes": [], "cumxg": []})
        mv = row["minute"]
        y_val = 0.0
        for m, xg in zip(td["minutes"], td["cumxg"]):
            if m <= mv:
                y_val = xg
        ax.scatter(
            mv,
            y_val,
            c=ann_color,
            s=350,
            zorder=8,
            edgecolors="white",
            lw=2.5,
            marker="*",
        )
        og_tag = f"  {OG_LABEL}" if is_og else ""
        scorer = _short(row["player"]) if row.get("player") else "?"
        ann_txt = f"⚽ {int(mv)}′  {scorer}{og_tag}"
        ax.annotate(
            ann_txt,
            xy=(mv, y_val),
            xytext=(12, 10),
            textcoords="offset points",
            color="white",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.42",
                facecolor=ann_color,
                alpha=0.92,
                edgecolor=OG_COLOR if is_og else "white",
                lw=2.2 if is_og else 0.7,
            ),
            arrowprops=dict(arrowstyle="-", color=ann_color, lw=1.2, alpha=0.8),
            zorder=9,
        )

    ax.text(
        0.99,
        0.97,
        s_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.45", facecolor=s_color, alpha=0.88, edgecolor="none"
        ),
        zorder=10,
    )

    names = list(xg_data.keys())
    if len(names) >= 2:
        for i, (name, col) in enumerate(zip(names, [C_RED, C_BLUE])):
            ax.text(
                0.01,
                0.97 - i * 0.11,
                f"  {name}   xG {xg_data[name]['xG']}  "
                f"| {xg_data[name]['goals']} G  "
                f"| {xg_data[name]['shots']} shots  ",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.42",
                    facecolor=col,
                    alpha=0.84,
                    edgecolor="none",
                ),
                zorder=10,
            )

    zone_patches = [
        mpatches.Patch(facecolor=zc, alpha=0.85, edgecolor=TEXT_DIM, lw=0.5, label=lbl)
        for _, _, code_, lbl, zc, _ in PERIOD_SPANS
        if code_ in periods_seen
    ]
    main_leg = ax.legend(
        loc="upper center",
        fontsize=10.5,
        ncol=2,
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        framealpha=0.90,
        bbox_to_anchor=(0.5, 1.02),
    )
    ax.add_artist(main_leg)
    if zone_patches:
        ax.legend(
            handles=zone_patches,
            loc="lower right",
            fontsize=8,
            ncol=len(zone_patches),
            facecolor=BG_MID,
            edgecolor=GRID_COL,
            labelcolor=TEXT_DIM,
            framealpha=0.85,
        )

    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.06, max(max_xg + 0.35, 1.2))
    ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=11, labelpad=6)
    ax.set_ylabel("Cumulative xG", color=TEXT_DIM, fontsize=11, labelpad=6)
    ax.tick_params(colors=TEXT_DIM, labelsize=10)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which="major", alpha=0.10, color=GRID_COL)
    ax.grid(which="minor", alpha=0.04, color=GRID_COL, linestyle=":")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(GRID_COL)


# ══════════════════════════════════════════════════════
#  FIG 2 & 3 — SHOT MAP
# ══════════════════════════════════════════════════════
def draw_shot_map_full(fig, events, team_id, team_name, team_color):
    fig.clear()
    fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.04, right=0.96)
    draw_pitch(ax)
    shots_df = events[events["is_shot"] == True].sort_values("minute")
    team_shots = shots_df[shots_df["team_id"] == team_id].copy()
    if team_shots.empty:
        ax.text(
            50,
            50,
            "No shots recorded",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=14,
            style="italic",
        )
        fig.text(
            0.50,
            0.975,
            f"Shot Map — {team_name}",
            ha="center",
            va="top",
            color=TEXT_BRIGHT,
            fontsize=15,
            fontweight="bold",
            transform=fig.transFigure,
        )
        return

    ax.add_patch(
        plt.Polygon(
            [[83.5, 21.1], [100, 21.1], [100, 78.9], [83.5, 78.9]],
            closed=True,
            facecolor="#ff000020",
            edgecolor="#ff000040",
            lw=1,
            zorder=1,
        )
    )

    legend_handles = []
    raw_col = (
        "shot_whoscored_type"
        if "shot_whoscored_type" in team_shots.columns
        else "shot_category"
    )
    for raw_type, (
        marker,
        face_col,
        edge_col,
        base_sz,
        zord,
        label,
    ) in SHOT_STYLE_RAW.items():
        subset = team_shots[team_shots[raw_col] == raw_type]
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            xg = row["xG"] or 0.0
            sz = base_sz + xg * (1400 if raw_type == "Goal" else 900)
            ax.scatter(
                row["x"],
                row["y"],
                c=face_col,
                s=sz,
                marker=marker,
                edgecolors=edge_col,
                linewidths=1.8,
                alpha=0.95,
                zorder=zord,
            )
            xg_str = f"xG {xg:.2f}"
            if raw_type == "Goal":
                is_og = row.get("is_own_goal", False)
                lbl_color = OG_COLOR if is_og else ACCENT_TEXT
                ax.text(
                    row["x"],
                    row["y"] - 7.5,
                    xg_str + (f"  {OG_LABEL}" if is_og else ""),
                    ha="center",
                    va="top",
                    color=lbl_color,
                    fontsize=9,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
                    zorder=zord + 1,
                )
                scorer = _short(row.get("player", ""))
                if scorer:
                    ax.text(
                        row["x"],
                        row["y"] + 9,
                        scorer,
                        ha="center",
                        va="bottom",
                        color=lbl_color,
                        fontsize=8.5,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="#000",
                            alpha=0.80,
                            edgecolor=lbl_color,
                            lw=1.0,
                        ),
                        zorder=zord + 1,
                    )
                ax.text(
                    row["x"] + 7,
                    row["y"] + 7,
                    f"{int(row['minute'])}'",
                    ha="left",
                    va="bottom",
                    color=_text_on_color(team_color),
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor=team_color,
                        alpha=0.88,
                        edgecolor="none",
                    ),
                    zorder=zord + 1,
                )
            else:
                ax.text(
                    row["x"],
                    row["y"] - 6,
                    xg_str,
                    ha="center",
                    va="top",
                    color=TEXT_BRIGHT,
                    fontsize=7.5,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="#000000",
                        alpha=0.70,
                        edgecolor=face_col,
                        lw=0.8,
                    ),
                    zorder=zord + 1,
                )
        legend_handles.append(
            mpatches.Patch(
                facecolor=face_col,
                edgecolor=edge_col,
                lw=1.5,
                label=f"{label} ({len(subset)})",
            )
        )

    ax.legend(
        handles=legend_handles,
        fontsize=10.5,
        ncol=3,
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        framealpha=0.95,
    )

    shot_counts = get_shot_counts(team_shots)
    tot_xg = round(float(team_shots["xG"].fillna(0).sum()), 2)
    big_ch = (
        int(team_shots["big_chance"].sum()) if "big_chance" in team_shots.columns else 0
    )

    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0, 1)
    _hdr.set_ylim(0, 1)
    _hdr.axis("off")
    _hdr.add_patch(
        plt.Rectangle((0, 0), 1.0, 1, facecolor=team_color, alpha=0.93, zorder=0)
    )
    _hdr.add_patch(
        plt.Rectangle((0, 0.82), 1.0, 0.18, facecolor="white", alpha=0.07, zorder=1)
    )
    _hdr.text(
        0.015,
        0.50,
        f"● {team_name}",
        ha="left",
        va="center",
        color=_text_on_color(team_color),
        fontsize=8.5,
        fontweight="bold",
        zorder=3,
    )
    _hdr.text(
        0.50,
        0.50,
        "Created by Mostafa Saad",
        ha="center",
        va="center",
        color=_accent_on_color(team_color),
        fontsize=8,
        fontweight="bold",
        fontstyle="italic",
        zorder=3,
    )

    fig.text(
        0.50,
        0.975,
        f"Shot Map — {team_name}",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=15,
        fontweight="bold",
        transform=fig.transFigure,
        path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
    )
    fig.text(
        0.50,
        0.951,
        (
            f"Total Shots: {shot_counts['shots']}     Goal: {shot_counts['goals']}     SavedShot: {shot_counts['saved']}"
            f"     MissedShots: {shot_counts['missed'] - shot_counts['post']}     BlockedShot: {shot_counts['blocked']}"
            f"     ShotOnPost: {shot_counts['post']}     xG: {tot_xg}     Big Chances: {big_ch}"
        ),
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=9,
        transform=fig.transFigure,
    )


# ══════════════════════════════════════════════════════
#  FIG 4 — BREAKDOWN + GOALS TABLE
# ══════════════════════════════════════════════════════
def draw_breakdown_goals(fig, events, info, xg_data):
    fig.clear()
    fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(
        2, 1, figure=fig, hspace=0.52, left=0.07, right=0.97, top=0.92, bottom=0.05
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1.set_facecolor(BG_MID)
    ax1.set_title(
        "WhoScored Shot Breakdown",
        color=TEXT_BRIGHT,
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    x_pos = np.arange(len(SHOT_BREAKDOWN_KEYS))
    w = 0.35
    for i, (name, color) in enumerate(zip(xg_data.keys(), [C_RED, C_BLUE])):
        vals = [xg_data[name].get(c, 0) for c in SHOT_BREAKDOWN_KEYS]
        bars = ax1.bar(
            x_pos + i * w,
            vals,
            w,
            label=name,
            color=color,
            alpha=0.88,
            edgecolor="white",
            lw=0.6,
        )
        _bar_txt = _text_on_color(color)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.06,
                    str(int(v) if float(v).is_integer() else round(v, 2)),
                    ha="center",
                    va="bottom",
                    color=TEXT_BRIGHT,
                    fontsize=10.5,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=BG_DARK)],
                )

    names_list = list(xg_data.keys())
    if len(names_list) == 2:
        xg_h = xg_data[names_list[0]]["xG"]
        xg_a = xg_data[names_list[1]]["xG"]
        total = xg_h + xg_a or 1
        ymax = max(ax1.get_ylim()[1], 5)
        y_bar = ymax * 0.92
        bh = ymax * 0.07
        ax1.barh(
            y_bar,
            xg_h / total * len(SHOT_BREAKDOWN_KEYS),
            height=bh,
            left=0,
            color=C_RED,
            alpha=0.85,
            zorder=5,
        )
        ax1.barh(
            y_bar,
            xg_a / total * len(SHOT_BREAKDOWN_KEYS),
            height=bh,
            left=xg_h / total * len(SHOT_BREAKDOWN_KEYS),
            color=C_BLUE,
            alpha=0.85,
            zorder=5,
        )
        _xg_h_txt = _text_on_color(C_RED)
        _xg_a_txt = _text_on_color(C_BLUE)
        ax1.text(
            0,
            y_bar,
            f" xG {xg_h}",
            va="center",
            color=_xg_h_txt,
            fontsize=10,
            fontweight="bold",
            zorder=6,
            path_effects=[
                pe.withStroke(
                    linewidth=2,
                    foreground="#000000" if _xg_h_txt == "#ffffff" else "#ffffff",
                )
            ],
        )
        ax1.text(
            len(SHOT_BREAKDOWN_KEYS),
            y_bar,
            f"xG {xg_a} ",
            va="center",
            ha="right",
            color=_xg_a_txt,
            fontsize=10,
            fontweight="bold",
            zorder=6,
            path_effects=[
                pe.withStroke(
                    linewidth=2,
                    foreground="#000000" if _xg_a_txt == "#ffffff" else "#ffffff",
                )
            ],
        )

    ax1.set_xticks(x_pos + w / 2)
    ax1.set_xticklabels(SHOT_BREAKDOWN_LABELS, color=TEXT_MAIN, fontsize=11)
    ax1.tick_params(colors=TEXT_MAIN, labelsize=10.5)
    ax1.legend(
        fontsize=11,
        facecolor=BG_DARK,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        framealpha=0.95,
    )
    ax1.grid(alpha=0.12, axis="y", color=GRID_COL)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax1.spines[sp].set_color(GRID_COL)
    ax1.text(
        0.5,
        -0.18,
        "On target = Goal + SavedShot | Off target = MissedShots + ShotOnPost",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=9,
    )

    ax2.clear()
    ax2.set_facecolor(BG_MID)
    ax2.axis("off")
    ax2.set_title(
        "Goals & Assists", color=TEXT_BRIGHT, fontsize=13, fontweight="bold", pad=10
    )
    gdf = events[events["is_goal"] == True].copy()
    if gdf.empty:
        ax2.text(
            0.5,
            0.5,
            "No goals recorded",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=14,
            style="italic",
            transform=ax2.transAxes,
        )
    else:
        gdf["Scorer By"] = gdf["team_id"].apply(
            lambda x: info["home_name"] if x == info["home_id"] else info["away_name"]
        )
        gdf["Scored For"] = gdf["scoring_team"].apply(
            lambda x: info["home_name"] if x == info["home_id"] else info["away_name"]
        )

        from match_report import (
            classify_goal_type as _classify_goal_type,
            goal_body_part_label as _goal_body_part_label,
        )

        def _goal_type_label(r):
            if r.get("is_own_goal", False):
                return "🔄 OG (Own Goal)"
            cat, sub = _classify_goal_type(r, events)
            body = _goal_body_part_label(r)
            if cat == "Penalty":
                return f"Penalty - {body}"
            if cat == "Set Piece":
                return f"Set Piece - {sub} - {body}"
            return f"Open Play - {body}"

        gdf["Type"] = gdf.apply(_goal_type_label, axis=1)
        gdf["Assist"] = gdf.apply(
            lambda r: (
                _short(str(r["assist_player"]))
                + (f" ({r['assist_type']})" if r["assist_type"] else "")
                if r["assist_player"]
                else "—"
            ),
            axis=1,
        )
        gdf["Scorer"] = gdf["player"].apply(_short)
        gdf["xG"] = gdf["xG"].apply(lambda v: f"{v:.3f}" if v else "—")
        gdf["Min"] = gdf["minute"].apply(lambda m: f"{int(m)}'" if pd.notna(m) else "—")
        disp = gdf[["Min", "Scorer", "Scorer By", "Scored For", "Type", "Assist", "xG"]]
        tbl = ax2.table(
            cellText=disp.values,
            colLabels=disp.columns.tolist(),
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10.5)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID_COL)
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#0A0A0A")
                cell.set_text_props(color=TEXT_BRIGHT, fontweight="bold", fontsize=11)
            else:
                row_data = disp.iloc[r - 1]
                is_og = "OWN GOAL" in str(row_data["Type"])
                if is_og:
                    cell.set_facecolor("#1e0a2e")
                    if c == 4:
                        cell.set_text_props(
                            color=OG_COLOR, fontweight="bold", fontsize=11
                        )
                    elif c in [1, 2, 3]:
                        cell.set_text_props(color="#e0aaff", fontweight="bold")
                    else:
                        cell.set_text_props(color="#c9b8e8")
                elif row_data["Scorer By"] == info["home_name"]:
                    cell.set_facecolor("#1a0a0a")
                    cell.set_text_props(color=TEXT_MAIN)
                else:
                    cell.set_facecolor("#050505")
                    cell.set_text_props(color=TEXT_MAIN)


# ══════════════════════════════════════════════════════
#  FIG 5 & 6 — PASS MAP
# ══════════════════════════════════════════════════════
def draw_pass_map_full(fig, events, team_id, team_name, team_color):
    fig.clear()
    fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_subplot(111)
    draw_pitch(ax, pitch_color=PITCH_COL)

    passes = events[
        (events["is_pass"] == True)
        & (events["team_id"] == team_id)
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if passes.empty:
        ax.text(
            50,
            50,
            "No pass data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=14,
            style="italic",
        )
        fig.text(
            0.50,
            0.975,
            f"Pass Map — {team_name}",
            ha="center",
            va="top",
            color=TEXT_BRIGHT,
            fontsize=15,
            fontweight="bold",
            transform=fig.transFigure,
        )
        return

    passes["zone"] = passes.apply(lambda r: _pass_zone(r["end_x"], r["end_y"]), axis=1)
    passes["successful"] = passes["outcome"] == "Successful"
    passes["is_key"] = passes["is_key_pass"] == True

    for zone, succ in [
        ("other", False),
        ("other", True),
        ("final_third", False),
        ("final_third", True),
        ("penalty", False),
        ("penalty", True),
    ]:
        sub = passes[
            ~passes["is_key"]
            & (passes["zone"] == zone)
            & (passes["successful"] == succ)
        ]
        if sub.empty:
            continue
        color, alpha, zord = _pass_color(zone, succ)
        for _, row in sub.iterrows():
            ax.annotate(
                "",
                xy=(row["end_x"], row["end_y"]),
                xytext=(row["x"], row["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#ffffff",
                    lw=2.8,
                    alpha=0.10,
                    mutation_scale=8,
                ),
                zorder=zord,
            )
            ax.annotate(
                "",
                xy=(row["end_x"], row["end_y"]),
                xytext=(row["x"], row["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.55,
                    alpha=alpha,
                    mutation_scale=8,
                ),
                zorder=zord + 1,
            )
    for _, row in passes[passes["is_key"]].iterrows():
        ax.annotate(
            "",
            xy=(row["end_x"], row["end_y"]),
            xytext=(row["x"], row["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color="#ffffff", lw=5.5, alpha=0.15, mutation_scale=12
            ),
            zorder=10,
        )
        ax.annotate(
            "",
            xy=(row["end_x"], row["end_y"]),
            xytext=(row["x"], row["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color="#facc15", lw=2.8, alpha=0.97, mutation_scale=11
            ),
            zorder=11,
        )
        ax.scatter(
            row["x"], row["y"], c="#facc15", s=55, edgecolors="white", lw=1.2, zorder=12
        )
        ax.scatter(
            row["end_x"],
            row["end_y"],
            c="#facc15",
            s=90,
            marker="*",
            edgecolors="white",
            lw=1.0,
            zorder=12,
        )
        ps = _short(row.get("player", ""))
        if ps:
            ax.text(
                row["end_x"],
                row["end_y"] + 3.5,
                ps,
                ha="center",
                va="bottom",
                color="#facc15",
                fontsize=8,
                fontweight="bold",
                zorder=13,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="#000000",
                    alpha=0.72,
                    edgecolor="#facc15",
                    lw=0.8,
                ),
            )

    ax.axvline(FINAL_THIRD_X, color="#94a3b8", lw=1.5, ls="--", alpha=0.55, zorder=6)
    ax.axvline(PENALTY_BOX_X, color="#94a3b8", lw=1.2, ls="--", alpha=0.45, zorder=6)

    total = len(passes)
    success = int(passes["successful"].sum())
    acc = round(success / total * 100, 1) if total else 0
    key_n = int(passes["is_key"].sum())
    pen_s = int(((passes["zone"] == "penalty") & passes["successful"]).sum())
    pen_f = int(((passes["zone"] == "penalty") & ~passes["successful"]).sum())
    ft_s = int(((passes["zone"] == "final_third") & passes["successful"]).sum())
    ft_f = int(((passes["zone"] == "final_third") & ~passes["successful"]).sum())
    xt_sum = (
        round(passes[passes["successful"] & passes["xT"].notna()]["xT"].sum(), 3)
        if "xT" in passes.columns
        else "—"
    )

    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor=C_BLUE,
                edgecolor="none",
                label=f"Successful — Final Third  ({ft_s})",
            ),
            mpatches.Patch(
                facecolor=C_GREEN,
                edgecolor="none",
                label=f"Successful — Penalty Box  ({pen_s})",
            ),
            mpatches.Patch(
                facecolor=C_RED,
                edgecolor="none",
                label=f"Failed — Final Third  ({ft_f})",
            ),
            mpatches.Patch(
                facecolor=C_GOLD,
                edgecolor="none",
                label=f"Failed — Penalty Box  ({pen_f})",
            ),
            mpatches.Patch(
                facecolor="#facc15",
                edgecolor="white",
                lw=1.0,
                label=f"⭐ Key Pass  ({key_n})",
            ),
        ],
        fontsize=10.5,
        ncol=3,
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        framealpha=0.95,
    )
    # ── Team colour bar ───────────────────────────────────────────
    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0, 1)
    _hdr.set_ylim(0, 1)
    _hdr.axis("off")
    _hdr.add_patch(
        plt.Rectangle((0, 0), 1.0, 1, facecolor=team_color, alpha=0.93, zorder=0)
    )
    _hdr.add_patch(
        plt.Rectangle((0, 0.82), 1.0, 0.18, facecolor="white", alpha=0.07, zorder=1)
    )
    _hdr.text(
        0.015,
        0.50,
        f"● {team_name}",
        ha="left",
        va="center",
        color=_text_on_color(team_color),
        fontsize=8.5,
        fontweight="bold",
        zorder=3,
    )
    _hdr.text(
        0.50,
        0.50,
        "Created by Mostafa Saad",
        ha="center",
        va="center",
        color=_accent_on_color(team_color),
        fontsize=8,
        fontweight="bold",
        fontstyle="italic",
        zorder=3,
    )
    fig.text(
        0.50,
        0.975,
        f"Pass Map — {team_name}",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=15,
        fontweight="bold",
        transform=fig.transFigure,
        path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
    )
    fig.text(
        0.50,
        0.951,
        f"Total: {total}   Completed: {success} ({acc}%)   "
        f"⭐ Key Passes: {key_n}   🎯 xT: {xt_sum}",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=9,
        transform=fig.transFigure,
    )


# ══════════════════════════════════════════════════════
#  FIG 7 & 8 — PASS NETWORK
# ══════════════════════════════════════════════════════
def draw_pass_network_full(
    fig, events, team_id, team_name, team_color, sub_in, sub_out, red_cards
):
    fig.clear()
    fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[4.5, 1],
        left=0.03,
        right=0.97,
        top=0.88,
        bottom=0.12,
        wspace=0.05,
    )
    ax_pitch = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.set_facecolor(BG_MID)
    ax_stats.axis("off")

    nodes, edges = build_pass_network(events, team_id)
    draw_pitch(ax_pitch, line_alpha=0.75)

    if not nodes or not edges:
        ax_pitch.text(
            50,
            50,
            "Not enough data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=14,
            style="italic",
        )
        fig.text(
            0.50,
            0.975,
            f"Pass Network — {team_name}",
            ha="center",
            va="top",
            color=TEXT_BRIGHT,
            fontsize=15,
            fontweight="bold",
            transform=fig.transFigure,
        )
        return

    max_edge = max(edges.values()) if edges else 1
    MIN_CONN = 3
    for (pid_a, pid_b), count in sorted(edges.items(), key=lambda x: x[1]):
        if count < MIN_CONN:
            continue
        if pid_a not in nodes or pid_b not in nodes:
            continue
        x1, y1 = nodes[pid_a]["avg_x"], nodes[pid_a]["avg_y"]
        x2, y2 = nodes[pid_b]["avg_x"], nodes[pid_b]["avg_y"]
        if any(np.isnan(v) for v in [x1, y1, x2, y2]):
            continue
        ratio = count / max_edge
        lw = 1.8 + ratio * 12
        alpha = 0.25 + ratio * 0.65
        edge_color = C_GREEN if ratio > 0.5 else C_GOLD
        ax_pitch.plot(
            [x1, x2],
            [y1, y2],
            color=edge_color,
            lw=lw + 6,
            alpha=alpha * 0.12,
            zorder=1,
            solid_capstyle="round",
        )
        ax_pitch.plot(
            [x1, x2],
            [y1, y2],
            color=edge_color,
            lw=lw,
            alpha=alpha,
            zorder=2,
            solid_capstyle="round",
        )
        if count >= MIN_CONN + 2:
            ax_pitch.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                str(count),
                ha="center",
                va="center",
                color=TEXT_BRIGHT,
                fontsize=8.5,
                fontweight="bold",
                zorder=4,
                bbox=dict(
                    boxstyle="round,pad=0.30",
                    facecolor=BG_DARK,
                    alpha=0.88,
                    edgecolor="none",
                ),
            )
    max_passes = max((n["pass_count"] for n in nodes.values()), default=1)
    for pid, node in nodes.items():
        if np.isnan(node["avg_x"]) or np.isnan(node["avg_y"]):
            continue
        sz = max(200, (node["pass_count"] / max_passes) * 1400)
        node_color = _player_role_color(pid, team_color, sub_in, sub_out, red_cards)
        badge = _player_role_badge(pid, sub_in, sub_out, red_cards)
        is_special = node_color != team_color
        ax_pitch.scatter(
            node["avg_x"], node["avg_y"], s=sz * 2.5, c=node_color, alpha=0.12, zorder=3
        )
        if is_special:
            ax_pitch.scatter(
                node["avg_x"],
                node["avg_y"],
                s=sz * 1.5,
                c="white",
                alpha=0.95,
                zorder=4,
            )
        ax_pitch.scatter(
            node["avg_x"],
            node["avg_y"],
            s=sz,
            c=node_color,
            zorder=5,
            edgecolors="white",
            lw=2.8,
            alpha=0.98,
        )
        ax_pitch.text(
            node["avg_x"],
            node["avg_y"],
            str(node["pass_count"]),
            ha="center",
            va="center",
            color=TEXT_BRIGHT,
            fontsize=9.5,
            fontweight="bold",
            zorder=7,
        )
        label_text = _short(node["name"]) + (f" {badge}" if badge else "")
        border_col = node_color if is_special else "#444444"
        name_color = (
            readable_team_text_color(node_color, BG_DARK) if is_special else TEXT_BRIGHT
        )
        ax_pitch.text(
            node["avg_x"],
            node["avg_y"] + 9,
            label_text,
            ha="center",
            va="bottom",
            color=name_color,
            fontsize=9,
            fontweight="bold",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="#000000",
                alpha=0.82,
                edgecolor=border_col,
                lw=1.5,
            ),
        )

    tp = events[(events["is_pass"] == True) & (events["team_id"] == team_id)]
    total_p = len(tp)
    success_p = int((tp["outcome"] == "Successful").sum())
    acc_p = round(success_p / total_p * 100, 1) if total_p else 0
    key_p = int(tp["is_key_pass"].sum())
    top_pairs = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:6]

    stats_lines = [
        ("PASS NETWORK", None, True),
        (team_name, team_color, True),
        ("", None, False),
        (f"Players:   {len(nodes)}", None, False),
        (f"Passes:    {total_p}", None, False),
        (f"Completed: {success_p}", None, False),
        (f"Accuracy:  {acc_p}%", None, False),
        (f"Key passes:{key_p}", None, False),
        ("", None, False),
        ("TOP PAIRS", None, True),
    ]
    for (pa, pb), cnt in top_pairs:
        na = _short(nodes[pa]["name"]) if pa in nodes else "?"
        nb = _short(nodes[pb]["name"]) if pb in nodes else "?"
        stats_lines.append((f"{na} ↔ {nb}  ({cnt})", None, False))

    y_pos = 0.97
    for text, color, bold in stats_lines:
        if text == "":
            y_pos -= 0.025
            continue
        fc = color if color else TEXT_MAIN
        ax_stats.text(
            0.08,
            y_pos,
            text,
            transform=ax_stats.transAxes,
            ha="left",
            va="top",
            color=fc,
            fontsize=10 if bold else 9.5,
            fontweight="bold" if bold else "normal",
        )
        y_pos -= 0.048 if bold else 0.042

    ax_pitch.legend(
        handles=[
            Line2D([0], [0], color=C_GOLD, lw=2, alpha=0.6, label="Low connection"),
            Line2D([0], [0], color=C_GREEN, lw=8, alpha=0.85, label="High connection"),
            mpatches.Patch(facecolor=team_color, edgecolor="white", label="Starter"),
            mpatches.Patch(
                facecolor=COLOR_SUB_IN, edgecolor="white", label="Sub In (↑)"
            ),
            mpatches.Patch(
                facecolor=COLOR_SUB_OUT, edgecolor="white", label="Subbed Off (↓)"
            ),
            mpatches.Patch(
                facecolor=COLOR_BOTH_SUB, edgecolor="white", label="Sub In+Off (↕)"
            ),
            mpatches.Patch(
                facecolor=COLOR_RED_CARD, edgecolor="white", label="Red Card (🟥)"
            ),
        ],
        fontsize=9,
        ncol=4,
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.92,
        borderpad=0.9,
        handlelength=2.0,
    )
    # ── Team colour bar ───────────────────────────────────────────
    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0, 1)
    _hdr.set_ylim(0, 1)
    _hdr.axis("off")
    _hdr.add_patch(
        plt.Rectangle((0, 0), 1.0, 1, facecolor=team_color, alpha=0.93, zorder=0)
    )
    _hdr.add_patch(
        plt.Rectangle((0, 0.82), 1.0, 0.18, facecolor="white", alpha=0.07, zorder=1)
    )
    _hdr.text(
        0.015,
        0.50,
        f"● {team_name}",
        ha="left",
        va="center",
        color=_text_on_color(team_color),
        fontsize=8.5,
        fontweight="bold",
        zorder=3,
    )
    _hdr.text(
        0.50,
        0.50,
        "Created by Mostafa Saad",
        ha="center",
        va="center",
        color=_accent_on_color(team_color),
        fontsize=8,
        fontweight="bold",
        fontstyle="italic",
        zorder=3,
    )
    fig.text(
        0.50,
        0.975,
        f"Pass Network — {team_name}",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=15,
        fontweight="bold",
        transform=fig.transFigure,
        path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
    )
    fig.text(
        0.50,
        0.951,
        f"Players: {len(nodes)}   Passes: {total_p}   Completed: {success_p} ({acc_p}%)   Key: {key_p}",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=9,
        transform=fig.transFigure,
    )


# ══════════════════════════════════════════════════════
#  FIG 9 & 10 — xT MAP
# ══════════════════════════════════════════════════════
def draw_xt_map_full(fig, events, team_id, team_name, team_color):
    fig.clear()
    fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[4.0, 1.1],
        left=0.05,
        right=0.96,
        top=0.87,
        bottom=0.10,
        wspace=0.06,
    )
    ax = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1])
    axs.set_facecolor(BG_MID)
    axs.axis("off")

    cmap = LinearSegmentedColormap.from_list(
        "xt", ["#050505", "#0A0A0A", "#1a6b3c", "#f59e0b", "#e63946", "#ff0044"]
    )

    grid_display = XT_GRID.T
    rows_n, cols_n = grid_display.shape
    cell_w = 100 / cols_n
    cell_h = 100 / rows_n

    im = ax.imshow(
        grid_display,
        extent=[0, 100, 0, 100],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=0.76,
        alpha=0.80,
        zorder=1,
    )

    x_centers = [(i + 0.5) * cell_w for i in range(cols_n)]
    y_centers = [(j + 0.5) * cell_h for j in range(rows_n)]
    for xi, xc in enumerate(x_centers):
        for yi, yc in enumerate(y_centers):
            val = grid_display[yi, xi]
            tc = "white" if val < 0.28 else "#111111"
            ax.text(
                xc,
                yc,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=tc,
                fontsize=7.8,
                fontweight="bold",
                zorder=3,
                path_effects=[
                    pe.withStroke(
                        linewidth=1.5, foreground="black" if tc == "white" else "white"
                    )
                ],
            )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.022, pad=0.015)
    cbar.set_label("xT Value", color="white", fontsize=10, labelpad=8)
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.outline.set_edgecolor("#3A3A3A")

    def pline(xs, ys, lw=1.5, alpha=0.75, ls="-"):
        ax.plot(
            xs,
            ys,
            color="white",
            lw=lw,
            alpha=alpha,
            ls=ls,
            zorder=4,
            solid_capstyle="round",
        )

    pline([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], lw=2.0)
    pline([50, 50], [0, 100], lw=1.2, alpha=0.50, ls="--")
    ax.add_patch(
        plt.Circle(
            (50, 50),
            9.15 * 0.68,
            color="white",
            fill=False,
            lw=1.0,
            alpha=0.40,
            zorder=4,
        )
    )
    pline(
        [83.5, 100, 100, 83.5, 83.5], [21.1, 21.1, 78.9, 78.9, 21.1], lw=1.1, alpha=0.50
    )
    pline([0, 16.5, 16.5, 0, 0], [21.1, 21.1, 78.9, 78.9, 21.1], lw=1.1, alpha=0.50)
    pline([94.5, 100, 100, 94.5], [36.8, 36.8, 63.2, 63.2], lw=0.9, alpha=0.40)
    pline([0, 5.5, 5.5, 0], [36.8, 36.8, 63.2, 63.2], lw=0.9, alpha=0.40)
    pline([100, 100], [44, 56], lw=4.0, alpha=0.92)
    pline([0, 0], [44, 56], lw=4.0, alpha=0.92)
    ax.axvline(FINAL_THIRD_X, color="white", lw=1.3, ls=":", alpha=0.40, zorder=4)
    ax.axvline(PENALTY_BOX_X, color="white", lw=1.0, ls=":", alpha=0.30, zorder=4)
    for txt, xp, yp, fs in [
        ("Defensive\nThird", 25, 98, 8.5),
        ("Middle\nThird", 58, 98, 8.5),
        ("Final\nThird", 83, 98, 8.5),
        ("Penalty\nBox", 91, 50, 8.0),
        ("← Attack Direction →", 50, -5.5, 9.5),
    ]:
        ax.text(
            xp,
            yp,
            txt,
            ha="center",
            va="top" if yp > 5 else "center",
            color="white",
            fontsize=fs,
            alpha=0.65,
            fontweight="bold",
            zorder=5,
        )
    ax.set_xlim(-2, 104)
    ax.set_ylim(-8, 104)
    ax.axis("off")

    passes = events[
        (events["is_pass"] == True)
        & (events["team_id"] == team_id)
        & (events["outcome"] == "Successful")
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if "xT" not in passes.columns or passes["xT"].isna().all():
        passes["xT"] = passes.apply(
            lambda r: calc_xt_pass(r["x"], r["y"], r["end_x"], r["end_y"]), axis=1
        )
    passes = passes[passes["xT"].notna()]
    xt_max = passes["xT"].abs().max() if not passes.empty else 1.0

    for _, row in passes.iterrows():
        val = row["xT"]
        if val >= 0:
            ratio = val / xt_max if xt_max else 0
            col_ = "white"
            lw_ = 0.5 + ratio * 2.6
            alpha = 0.12 + ratio * 0.72
        else:
            ratio = abs(val) / xt_max if xt_max else 0
            col_ = "#ff6b6b"
            lw_ = 0.5 + ratio * 1.2
            alpha = 0.10 + ratio * 0.38
        ax.annotate(
            "",
            xy=(row["end_x"], row["end_y"]),
            xytext=(row["x"], row["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color=col_, lw=lw_, alpha=alpha, mutation_scale=6
            ),
            zorder=6,
        )

    if not passes.empty:
        top5 = passes.nlargest(5, "xT")
        used = []
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            ax.annotate(
                "",
                xy=(row["end_x"], row["end_y"]),
                xytext=(row["x"], row["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#7DD3FC",
                    lw=5.5,
                    alpha=0.15,
                    mutation_scale=14,
                ),
                zorder=7,
            )
            ax.annotate(
                "",
                xy=(row["end_x"], row["end_y"]),
                xytext=(row["x"], row["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#7DD3FC",
                    lw=2.8,
                    alpha=0.96,
                    mutation_scale=12,
                ),
                zorder=8,
            )
            ax.scatter(
                row["x"],
                row["y"],
                c="#7DD3FC",
                s=45,
                edgecolors="black",
                lw=0.8,
                zorder=9,
            )
            ax.scatter(
                row["end_x"],
                row["end_y"],
                c="#7DD3FC",
                s=180,
                marker="*",
                edgecolors="white",
                lw=1.2,
                zorder=9,
            )
            lbl_y = row["end_y"] + 4.5
            for px, py in used:
                if abs(row["end_x"] - px) < 12 and abs(lbl_y - py) < 6:
                    lbl_y += 6
            used.append((row["end_x"], lbl_y))
            ax.text(
                row["end_x"],
                lbl_y,
                f"#{rank} {_short(row.get('player',''))}  +{row['xT']:.3f}",
                ha="center",
                va="bottom",
                color="#7DD3FC",
                fontsize=8.2,
                fontweight="bold",
                zorder=10,
                bbox=dict(
                    boxstyle="round,pad=0.30",
                    facecolor="#050508",
                    alpha=0.85,
                    edgecolor="#7DD3FC",
                    lw=1.0,
                ),
            )

    xt_total = round(passes["xT"].sum(), 3) if not passes.empty else 0.0
    xt_best = round(passes["xT"].max(), 3) if not passes.empty else 0.0
    n_pos = int((passes["xT"] > 0).sum()) if not passes.empty else 0
    n_neg = int((passes["xT"] < 0).sum()) if not passes.empty else 0
    n_total = len(passes)
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)

    def card(y0, height, color, alpha=0.18):
        axs.add_patch(
            mpatches.FancyBboxPatch(
                (0.05, y0),
                0.90,
                height,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                lw=1.2,
                transform=axs.transAxes,
                zorder=1,
            )
        )

    def divider(y):
        axs.plot(
            [0.05, 0.95],
            [y, y],
            color="#2d3748",
            lw=1.0,
            transform=axs.transAxes,
            zorder=2,
        )

    def lbl(txt, x, y, color=TEXT_MAIN, size=9, bold=False, ha="left"):
        axs.text(
            x,
            y,
            txt,
            transform=axs.transAxes,
            ha=ha,
            va="center",
            color=color,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            clip_on=False,
            zorder=3,
        )

    card(0.91, 0.08, team_color, alpha=0.25)
    lbl("⚡  xT MAP", 0.50, 0.96, team_color, size=11, bold=True, ha="center")
    lbl(team_name, 0.50, 0.92, team_color, size=9, bold=True, ha="center")
    divider(0.90)
    card(0.78, 0.11, team_color, alpha=0.20)
    lbl("TOTAL  xT", 0.50, 0.875, TEXT_DIM, size=8, ha="center")
    lbl(f"{xt_total:+.4f}", 0.50, 0.820, TEXT_BRIGHT, size=16, bold=True, ha="center")
    divider(0.77)
    yq = 0.74
    for label, val, col in [
        ("Best Pass", f"+{xt_best:.3f}", TEXT_MAIN),
        ("Total Passes", str(n_total), TEXT_MAIN),
        ("Positive xT", str(n_pos), "#22c55e"),
        ("Negative xT", str(n_neg), "#e63946"),
    ]:
        lbl(label, 0.08, yq, TEXT_DIM, size=8.5)
        lbl(val, 0.92, yq, col, size=9, bold=True, ha="right")
        yq -= 0.052
    divider(yq + 0.010)
    yq -= 0.010
    lbl("TOP  xT  CREATORS", 0.50, yq, TEXT_DIM, size=8.5, bold=True, ha="center")
    yq -= 0.045
    if not passes.empty:
        top_p = (
            passes[passes["xT"] > 0]
            .groupby("player")["xT"]
            .sum()
            .sort_values(ascending=False)
            .head(7)
        )
        mx = top_p.max() if not top_p.empty else 1.0
        for rank, (player, xval) in enumerate(top_p.items(), 1):
            bw = max(min(xval / mx * 0.78, 0.78), 0.04)
            bar_col = team_color if rank > 1 else "#7DD3FC"
            axs.add_patch(
                mpatches.FancyBboxPatch(
                    (0.08, yq - 0.018),
                    0.84,
                    0.036,
                    boxstyle="round,pad=0.005",
                    facecolor="#0A0A0A",
                    edgecolor="none",
                    transform=axs.transAxes,
                    zorder=2,
                )
            )
            axs.add_patch(
                mpatches.FancyBboxPatch(
                    (0.08, yq - 0.018),
                    bw * 0.84,
                    0.036,
                    boxstyle="round,pad=0.005",
                    facecolor=bar_col,
                    edgecolor="none",
                    alpha=0.45,
                    transform=axs.transAxes,
                    zorder=3,
                )
            )
            name_col = "#7DD3FC" if rank == 1 else TEXT_BRIGHT
            lbl(
                f"{rank}. {_short(player)}",
                0.10,
                yq,
                name_col,
                size=8.2,
                bold=(rank == 1),
            )
            lbl(
                f"+{xval:.3f}",
                0.90,
                yq,
                "#7DD3FC" if rank == 1 else TEXT_DIM,
                size=8.2,
                bold=(rank == 1),
                ha="right",
            )
            yq -= 0.060

    ax.legend(
        handles=[
            mpatches.Patch(facecolor="white", alpha=0.80, label="Positive xT pass"),
            mpatches.Patch(facecolor="#ff6b6b", alpha=0.80, label="Negative xT pass"),
            mpatches.Patch(
                facecolor="#7DD3FC", edgecolor="white", lw=1, label="Top-5 xT passes"
            ),
        ],
        fontsize=9.5,
        ncol=3,
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        framealpha=0.92,
    )
    # ── Team colour bar ───────────────────────────────────────────
    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0, 1)
    _hdr.set_ylim(0, 1)
    _hdr.axis("off")
    _hdr.add_patch(
        plt.Rectangle((0, 0), 1.0, 1, facecolor=team_color, alpha=0.93, zorder=0)
    )
    _hdr.add_patch(
        plt.Rectangle((0, 0.82), 1.0, 0.18, facecolor="white", alpha=0.07, zorder=1)
    )
    _hdr.text(
        0.015,
        0.50,
        f"● {team_name}",
        ha="left",
        va="center",
        color=_text_on_color(team_color),
        fontsize=8.5,
        fontweight="bold",
        zorder=3,
    )
    _hdr.text(
        0.50,
        0.50,
        "Created by Mostafa Saad",
        ha="center",
        va="center",
        color=_accent_on_color(team_color),
        fontsize=8,
        fontweight="bold",
        fontstyle="italic",
        zorder=3,
    )
    fig.text(
        0.50,
        0.975,
        f"xT Map — {team_name}",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=15,
        fontweight="bold",
        transform=fig.transFigure,
        path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
    )
    fig.text(
        0.50,
        0.951,
        f"Total xT: {xt_total:+.3f}   ⭐ Best Pass: +{xt_best:.3f}   Passes: {n_total}",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=9,
        transform=fig.transFigure,
    )


# ══════════════════════════════════════════════════════
#  FIG 11 — MATCH REPORT helpers
# ══════════════════════════════════════════════════════
def _mini_pitch(ax, bg=PITCH_COL):
    """
    Draw a horizontal mini pitch with subtle visual enhancements:
      • Faint penalty-box fill (team actions plotted on top)
      • Slightly brighter centre circle
      • Goal posts with cap style
    """
    ax.set_facecolor(bg)

    def l(xs, ys, lw=1.2, a=0.70, ls="-"):
        ax.plot(
            xs,
            ys,
            color="white",
            lw=lw,
            alpha=a,
            ls=ls,
            zorder=2,
            solid_capstyle="round",
        )

    # ── Subtle box fills ──────────────────────────────────────────
    ax.add_patch(
        plt.Rectangle(
            (83.5, 21.1),
            16.5,
            57.8,
            facecolor="#ffffff",
            alpha=0.025,
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (0, 21.1),
            16.5,
            57.8,
            facecolor="#ffffff",
            alpha=0.018,
            edgecolor="none",
            zorder=1,
        )
    )
    # ── Pitch outline & markings ──────────────────────────────────
    l([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], lw=1.9, a=0.88)
    l([50, 50], [0, 100], lw=1.0, a=0.40, ls="--")
    ax.add_patch(
        plt.Circle(
            (50, 50),
            9.15 * 0.68,
            color="white",
            fill=False,
            lw=0.9,
            alpha=0.38,
            zorder=2,
        )
    )
    ax.plot(50, 50, "o", color="white", ms=2.0, alpha=0.55, zorder=2)
    l([83.5, 100, 100, 83.5, 83.5], [21.1, 21.1, 78.9, 78.9, 21.1], lw=1.0, a=0.55)
    l([0, 16.5, 16.5, 0, 0], [21.1, 21.1, 78.9, 78.9, 21.1], lw=1.0, a=0.55)
    l([94.5, 100, 100, 94.5], [36.8, 36.8, 63.2, 63.2], lw=0.75, a=0.42)
    l([0, 5.5, 5.5, 0], [36.8, 36.8, 63.2, 63.2], lw=0.75, a=0.42)
    # Goal posts — brighter
    l([100, 100], [44, 56], lw=4.0, a=0.96)
    l([0, 0], [44, 56], lw=4.0, a=0.96)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-8, 104)
    ax.axis("off")


def _lbl(ax, txt, col=TEXT_BRIGHT, size=8.5):
    """
    Title above any axes — uses transAxes so it works for ALL panel types.
    Enhanced with glow path-effect and sharper badge.
    """
    ax.text(
        0.50,
        1.028,
        txt,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        color=col,
        fontsize=size,
        fontweight="bold",
        zorder=20,
        clip_on=False,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="#000000")],
        bbox=dict(
            boxstyle="round,pad=0.40",
            facecolor="#07090f",
            edgecolor=col,
            linewidth=1.4,
            alpha=0.97,
        ),
    )


def _rpt_pass_network(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Pass Network — {name}", tc)
    p = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if p.empty:
        return
    avg = p.groupby("player")[["x", "y"]].mean()
    cnt = p.groupby("player").size().rename("n")
    avg = avg.join(cnt)
    mx_n = avg["n"].max() if not avg.empty else 1
    for pl, row in avg.iterrows():
        s = 18 + row["n"] / mx_n * 85
        ax.scatter(row["x"], row["y"], c=tc, s=s, edgecolors="white", lw=0.7, zorder=5)
        nm = pl.split()[-1][:7] if pl else ""
        ax.text(
            row["x"],
            row["y"] + 3.5,
            nm,
            ha="center",
            va="bottom",
            color="white",
            fontsize=5.5,
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#000000",
                alpha=0.70,
                edgecolor="none",
            ),
        )


def _rpt_shot_table(ax, events, info, xg_data):
    ax.set_facecolor(BG_MID)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hn, an = info["home_name"], info["away_name"]
    hd, ad = xg_data.get(hn, {}), xg_data.get(an, {})
    hs = events[(events["is_shot"] == True) & (events["team_id"] == info["home_id"])]
    as_ = events[(events["is_shot"] == True) & (events["team_id"] == info["away_id"])]
    hxgot = round(
        hs[
            (
                hs.get("shot_whoscored_type", hs["shot_category"]).isin(
                    ["Goal", "SavedShot", "On Target"]
                )
            )
        ]["xG"].sum(),
        2,
    )
    axgot = round(
        as_[
            (
                as_.get("shot_whoscored_type", as_["shot_category"]).isin(
                    ["Goal", "SavedShot", "On Target"]
                )
            )
        ]["xG"].sum(),
        2,
    )
    rows_ = [
        ("Goals", hd.get("goals", 0), ad.get("goals", 0), C_GOLD),
        ("xG", hd.get("xG", 0), ad.get("xG", 0), "#a855f7"),
        ("xGoT", hxgot, axgot, "#F5C542"),
        ("Shots", hd.get("shots", 0), ad.get("shots", 0), "#94a3b8"),
        ("On Tgt", hd.get("on_target", 0), ad.get("on_target", 0), C_GREEN),
        ("Blocked", hd.get("blocked", 0), ad.get("blocked", 0), C_GOLD),
        ("Off Target", hd.get("missed", 0), ad.get("missed", 0), "#64748b"),
        ("Big Ch.", hd.get("big_chances", 0), ad.get("big_chances", 0), "#f43f5e"),
    ]
    ax.text(
        0.10,
        0.97,
        hn[:10],
        ha="left",
        va="top",
        color=C_RED,
        fontsize=8.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.97,
        "SHOTS",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=7.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.90,
        0.97,
        an[:10],
        ha="right",
        va="top",
        color=C_BLUE,
        fontsize=8.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.plot([0.03, 0.97], [0.93, 0.93], color=GRID_COL, lw=0.8, transform=ax.transAxes)
    y = 0.875
    for lbl_, hv, av, col in rows_:
        tot = (float(hv) + float(av)) or 1
        hr = float(hv) / tot
        ax.barh(
            y,
            hr * 0.22,
            height=0.055,
            left=0.03,
            color=C_RED,
            alpha=0.55,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.barh(
            y,
            (1 - hr) * 0.22,
            height=0.055,
            left=0.75,
            color=C_BLUE,
            alpha=0.55,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.text(
            0.27,
            y,
            str(hv),
            ha="right",
            va="center",
            color=C_RED,
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.50,
            y,
            lbl_,
            ha="center",
            va="center",
            color=col,
            fontsize=7.5,
            fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=BG_DARK,
                alpha=0.85,
                edgecolor=col,
                lw=0.8,
            ),
        )
        ax.text(
            0.73,
            y,
            str(av),
            ha="left",
            va="center",
            color=C_BLUE,
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        y -= 0.102
    _lbl(ax, "Shot Statistics", TEXT_DIM)


def _rpt_avg_position(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Avg Positions — {name}", tc)
    ev = events[
        (events["team_id"] == tid) & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if ev.empty:
        return
    avg = ev.groupby("player")[["x", "y"]].mean()
    cnt = ev.groupby("player").size()
    mx = cnt.max() if not cnt.empty else 1
    for pl, row in avg.iterrows():
        n = cnt.get(pl, 1)
        ax.scatter(
            row["x"],
            row["y"],
            c=tc,
            s=25 + n / mx * 80,
            edgecolors="white",
            lw=0.7,
            alpha=0.85,
            zorder=4,
        )
        nm = pl.split()[-1][:6] if pl else ""
        ax.text(
            row["x"],
            row["y"] + 3.2,
            nm,
            ha="center",
            va="bottom",
            color="white",
            fontsize=5.2,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="#000000",
                alpha=0.68,
                edgecolor="none",
            ),
        )


def _rpt_gk_saves(ax, events, info):
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    _lbl(ax, "Goalkeeper Saves", TEXT_DIM)

    def goal_box(cx, cy, w, h, col, lbl_, n):
        ax.plot(
            [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
            [cy, cy, cy + h, cy + h, cy],
            color="white",
            lw=1.5,
            alpha=0.8,
            zorder=3,
        )
        ax.text(
            cx,
            cy - 6,
            f"{lbl_[:9]}\n{n} saves",
            ha="center",
            va="top",
            color=col,
            fontsize=7,
            fontweight="bold",
        )

    goal_box(
        25,
        30,
        36,
        30,
        C_RED,
        info["home_name"],
        len(
            events[
                (events["team_id"] == info["away_id"])
                & (events["shot_category"] == "On Target")
            ]
        ),
    )
    goal_box(
        75,
        30,
        36,
        30,
        C_BLUE,
        info["away_name"],
        len(
            events[
                (events["team_id"] == info["home_id"])
                & (events["shot_category"] == "On Target")
            ]
        ),
    )
    for tid, cx, col in [(info["away_id"], 25, C_RED), (info["home_id"], 75, C_BLUE)]:
        sv = events[
            (events["team_id"] == tid)
            & (events["shot_category"] == "On Target")
            & events[["end_x", "end_y"]].notna().all(axis=1)
        ]
        if sv.empty:
            continue
        sx = cx + (sv["end_y"] - 50) * 0.36
        sy = 30 + (sv["end_x"].clip(83.5, 100) - 83.5) / 16.5 * 30
        ax.scatter(
            sx, sy, c=col, s=55, edgecolors="white", lw=0.8, alpha=0.88, zorder=5
        )


def _rpt_progressive(ax, events, tid, tc, name):
    _mini_pitch(ax)
    p = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    prog = p[(p["end_x"] - p["x"]) >= 10].copy() if not p.empty else p
    n = len(prog)
    _lbl(ax, f"{name}: {n} Progressive Passes", tc)
    if prog.empty:
        return

    def zc(x):
        if x < 33:
            return "#64748b"
        if x < 66:
            return C_GOLD
        return C_GREEN

    for _, r in prog.iterrows():
        ax.annotate(
            "",
            xy=(r["end_x"], r["end_y"]),
            xytext=(r["x"], r["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color=zc(r["x"]), lw=0.9, alpha=0.55, mutation_scale=5
            ),
            zorder=3,
        )
    n1 = int((prog["x"] < 33).sum())
    n2 = int(((prog["x"] >= 33) & (prog["x"] < 66)).sum())
    n3 = int((prog["x"] >= 66).sum())
    for i, (lv, cl) in enumerate(
        [
            (f"Own 3rd  {n1}({int(n1/n*100) if n else 0}%)", "#64748b"),
            (f"Middle   {n2}({int(n2/n*100) if n else 0}%)", C_GOLD),
            (f"Final 3rd {n3}({int(n3/n*100) if n else 0}%)", C_GREEN),
        ]
    ):
        ax.text(
            1,
            99 - i * 9,
            lv,
            ha="right",
            va="top",
            color=cl,
            fontsize=6.5,
            fontweight="bold",
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=BG_DARK,
                alpha=0.80,
                edgecolor="none",
            ),
        )


def _rpt_xt_minute(ax, events, info):
    ax.set_facecolor(BG_DARK)
    _lbl(ax, "xT — Match Dominance per Minute", TEXT_DIM)
    if "xT" not in events.columns:
        ax.text(
            0.5,
            0.5,
            "No xT data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            transform=ax.transAxes,
        )
        return
    xt = events[
        (events["xT"].notna())
        & (events["xT"] > 0)
        & (events["outcome"] == "Successful")
    ].copy()
    hxt = xt[xt["team_id"] == info["home_id"]].groupby("minute")["xT"].sum()
    axt = xt[xt["team_id"] == info["away_id"]].groupby("minute")["xT"].sum()
    mins = list(range(1, 96))
    ax.bar(mins, [hxt.get(m, 0) for m in mins], color=C_RED, alpha=0.75, width=0.8)
    ax.bar(mins, [-axt.get(m, 0) for m in mins], color=C_BLUE, alpha=0.75, width=0.8)
    ax.axhline(0, color="white", lw=0.8, alpha=0.5)
    for xp, lb in [(45, "HT"), (90, "FT")]:
        ax.axvline(xp, color=C_GOLD, lw=1.0, ls="--", alpha=0.60)
        ax.text(xp + 0.5, ax.get_ylim()[1] * 0.85, lb, color=TEXT_BRIGHT, fontsize=6)
    ht = round(xt[xt["team_id"] == info["home_id"]]["xT"].sum(), 3)
    at = round(xt[xt["team_id"] == info["away_id"]]["xT"].sum(), 3)
    for pos, name, col, ha_ in [
        (0.02, info["home_name"], C_RED, "left"),
        (0.98, info["away_name"], C_BLUE, "right"),
    ]:
        ax.text(
            pos,
            0.97,
            f"{name[:10]}  xT:{ht if col==C_RED else at}",
            transform=ax.transAxes,
            ha=ha_,
            va="top",
            color=col,
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=BG_MID,
                alpha=0.85,
                edgecolor="none",
            ),
        )
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=7, labelpad=2)
    ax.set_ylabel("xT / min", color=TEXT_DIM, fontsize=7, labelpad=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(GRID_COL)
    ax.grid(alpha=0.07, color=GRID_COL)


def _rpt_zone14(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Zone 14 & Half-Spaces — {name}", tc)
    ev = events[
        (events["team_id"] == tid) & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    from matplotlib.patches import Rectangle as Rect

    ax.add_patch(
        Rect((66, 33), 17, 34, facecolor=tc, alpha=0.22, edgecolor=tc, lw=1.5, zorder=2)
    )
    ax.add_patch(
        Rect(
            (66, 67),
            17,
            13,
            facecolor="#a855f7",
            alpha=0.22,
            edgecolor="#a855f7",
            lw=1.2,
            zorder=2,
        )
    )
    ax.add_patch(
        Rect(
            (66, 20),
            17,
            13,
            facecolor="#a855f7",
            alpha=0.22,
            edgecolor="#a855f7",
            lw=1.2,
            zorder=2,
        )
    )
    if ev.empty:
        return
    z14 = ev[ev["x"].between(66, 83) & ev["y"].between(33, 67)]
    lhs = ev[ev["x"].between(66, 83) & ev["y"].between(67, 80)]
    rhs = ev[ev["x"].between(66, 83) & ev["y"].between(20, 33)]
    if not z14.empty:
        ax.scatter(
            z14["x"], z14["y"], c=tc, s=12, alpha=0.60, zorder=4, edgecolors="none"
        )
    for val, yx, yy, col in [
        (len(z14), 74.5, 50, tc),
        (len(lhs), 74.5, 73.5, "#a855f7"),
        (len(rhs), 74.5, 26.5, "#a855f7"),
    ]:
        ax.text(
            yx,
            yy,
            str(val),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            zorder=6,
            bbox=dict(
                boxstyle="circle,pad=0.35",
                facecolor=col,
                alpha=0.88,
                edgecolor="white",
                lw=0.9,
            ),
        )
    ax.text(
        50,
        -4.5,
        f"Zone14:{len(z14)}  L.Half-Space:{len(lhs)}  R.Half-Space:{len(rhs)}",
        ha="center",
        va="top",
        color=tc,
        fontsize=6.5,
        fontweight="bold",
    )


def _rpt_stats_table(ax, events, info, xg_data):
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]
    hd, ad = xg_data.get(hn, {}), xg_data.get(an, {})
    hp = int(events[(events["is_pass"] == True) & (events["team_id"] == hid)].shape[0])
    ap = int(events[(events["is_pass"] == True) & (events["team_id"] == aid)].shape[0])
    tot_p = (hp + ap) or 1
    hps = int(
        events[
            (events["is_pass"] == True)
            & (events["team_id"] == hid)
            & (events["outcome"] == "Successful")
        ].shape[0]
    )
    aps = int(
        events[
            (events["is_pass"] == True)
            & (events["team_id"] == aid)
            & (events["outcome"] == "Successful")
        ].shape[0]
    )
    hkp = int(
        events[(events["is_key_pass"] == True) & (events["team_id"] == hid)].shape[0]
    )
    akp = int(
        events[(events["is_key_pass"] == True) & (events["team_id"] == aid)].shape[0]
    )
    if "xT" in events.columns:
        hxt = round(
            events[(events["team_id"] == hid) & events["xT"].notna()]["xT"].sum(), 2
        )
        axt = round(
            events[(events["team_id"] == aid) & events["xT"].notna()]["xT"].sum(), 2
        )
    else:
        hxt, axt = 0, 0
    stats = [
        (
            "Possession",
            f"{round(hp/tot_p*100,1)}%",
            f"{round(ap/tot_p*100,1)}%",
            hp / tot_p,
            C_GOLD,
        ),
        (
            "Passes (Acc)",
            f"{hp}({hps})",
            f"{ap}({aps})",
            hp / ((hp + ap) or 1),
            "#94a3b8",
        ),
        (
            "Shots (OnTgt)",
            f"{hd.get('shots',0)}({hd.get('on_target',0)})",
            f"{ad.get('shots',0)}({ad.get('on_target',0)})",
            hd.get("shots", 0) / ((hd.get("shots", 0) + ad.get("shots", 0)) or 1),
            "#64748b",
        ),
        (
            "xG",
            str(hd.get("xG", 0)),
            str(ad.get("xG", 0)),
            hd.get("xG", 0) / ((hd.get("xG", 0) + ad.get("xG", 0)) or 1),
            "#a855f7",
        ),
        ("xT", str(hxt), str(axt), hxt / ((hxt + axt) or 1), "#22c55e"),
        ("Key Passes", str(hkp), str(akp), hkp / ((hkp + akp) or 1), "#F5C542"),
        (
            "Big Chances",
            str(hd.get("big_chances", 0)),
            str(ad.get("big_chances", 0)),
            hd.get("big_chances", 0)
            / ((hd.get("big_chances", 0) + ad.get("big_chances", 0)) or 1),
            "#f43f5e",
        ),
    ]
    ax.text(
        0.50,
        0.99,
        "MATCH STATISTICS",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=9.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.10,
        0.94,
        hn[:12],
        ha="left",
        va="top",
        color=C_RED,
        fontsize=8,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.90,
        0.94,
        an[:12],
        ha="right",
        va="top",
        color=C_BLUE,
        fontsize=8,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.plot([0.02, 0.98], [0.90, 0.90], color=GRID_COL, lw=0.8, transform=ax.transAxes)
    y = 0.85
    step = 0.118
    for lbl_, hv, av, hr, col in stats:
        ax.add_patch(
            plt.Rectangle(
                (0.02, y - 0.048),
                hr * 0.96,
                0.050,
                facecolor=C_RED,
                alpha=0.18,
                transform=ax.transAxes,
                zorder=1,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (0.02 + hr * 0.96, y - 0.048),
                (1 - hr) * 0.96,
                0.050,
                facecolor=C_BLUE,
                alpha=0.18,
                transform=ax.transAxes,
                zorder=1,
            )
        )
        ax.text(
            0.06,
            y,
            str(hv),
            ha="left",
            va="center",
            color=C_RED,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.50,
            y,
            lbl_,
            ha="center",
            va="center",
            color=col,
            fontsize=7.8,
            fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=BG_MID,
                alpha=0.92,
                edgecolor=col,
                lw=0.8,
            ),
        )
        ax.text(
            0.94,
            y,
            str(av),
            ha="right",
            va="center",
            color=C_BLUE,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
        )
        y -= step


def _rpt_pass_zones(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Pass Zones — {name}", tc)
    p = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if p.empty:
        return
    tot = len(p)
    col_e = [0, 20, 40, 60, 80, 100]
    row_e = [0, 34, 67, 100]
    for ci in range(5):
        for ri in range(3):
            x0, x1 = col_e[ci], col_e[ci + 1]
            y0, y1 = row_e[ri], row_e[ri + 1]
            n = int(
                p[
                    (p["x"] >= x0) & (p["x"] < x1) & (p["y"] >= y0) & (p["y"] < y1)
                ].shape[0]
            )
            if n == 0:
                continue
            pct = round(n / tot * 100, 0)
            inten = min(pct / 12, 1.0)
            ax.add_patch(
                plt.Rectangle(
                    (x0 + 0.5, y0 + 0.5),
                    x1 - x0 - 1,
                    y1 - y0 - 1,
                    facecolor=tc,
                    alpha=inten * 0.58,
                    edgecolor="none",
                    zorder=2,
                )
            )
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                f"{int(pct)}%",
                ha="center",
                va="center",
                color=_text_on_color(tc),
                fontsize=7.5,
                fontweight="bold",
                zorder=3,
                path_effects=[
                    pe.withStroke(linewidth=1.8, foreground=_stroke_on_color(tc))
                ],
            )


def _rpt_crosses(ax, events, info):
    _mini_pitch(ax)
    _lbl(ax, "Crosses", TEXT_DIM)
    if "is_cross" not in events.columns:
        ax.text(
            50,
            50,
            "No cross data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return
    crs = events[
        (events["is_cross"] == True)
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if crs.empty:
        ax.text(
            50,
            50,
            "No crosses recorded",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return
    for tid, col in [(info["home_id"], C_RED), (info["away_id"], C_BLUE)]:
        tc = crs[crs["team_id"] == tid]
        for _, r in tc.iterrows():
            succ = r.get("outcome", "") == "Successful"
            ax.annotate(
                "",
                xy=(r["end_x"], r["end_y"]),
                xytext=(r["x"], r["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=1.3 if succ else 0.6,
                    alpha=0.82 if succ else 0.30,
                    mutation_scale=6,
                ),
                zorder=4,
            )
        eff = int(tc[tc["outcome"] == "Successful"].shape[0])
        nm = info["home_name"][:8] if tid == info["home_id"] else info["away_name"][:8]
        xp = 2 if tid == info["home_id"] else 98
        ax.text(
            xp,
            -4.5,
            f"{nm}: {len(tc)} ({eff} eff.)",
            ha="left" if tid == info["home_id"] else "right",
            va="top",
            color=col,
            fontsize=6.5,
            fontweight="bold",
        )


def _rpt_danger_zones(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Danger Creation — {name}", tc)
    dng = events[
        ((events["is_shot"] == True) | (events["is_key_pass"] == True))
        & (events["team_id"] == tid)
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if dng.empty:
        return
    shots = dng[dng["is_shot"] == True]
    kp = dng[dng["is_key_pass"] == True]
    goals = shots[shots["is_goal"] == True] if not shots.empty else shots
    if not dng.empty:
        ax.scatter(
            dng["x"], dng["y"], c=tc, s=120, alpha=0.08, zorder=2, edgecolors="none"
        )
        ax.scatter(
            dng["x"], dng["y"], c=tc, s=40, alpha=0.12, zorder=2, edgecolors="none"
        )
    if not kp.empty:
        ax.scatter(
            kp["x"],
            kp["y"],
            c="#facc15",
            s=22,
            marker="^",
            alpha=0.78,
            zorder=4,
            edgecolors="none",
        )
    if not shots.empty:
        ax.scatter(
            shots["x"],
            shots["y"],
            c=tc,
            s=28,
            alpha=0.75,
            zorder=4,
            edgecolors="white",
            lw=0.5,
        )
    if not goals.empty:
        ax.scatter(
            goals["x"],
            goals["y"],
            c="#FFD700",
            s=90,
            marker="*",
            zorder=6,
            edgecolors="white",
            lw=0.8,
        )
    ax.text(
        50,
        -4.5,
        f"Shots:{len(shots)}  Key Pass:{len(kp)}  Goals:{len(goals)}",
        ha="center",
        va="top",
        color=tc,
        fontsize=6.5,
        fontweight="bold",
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#facc15",
                markersize=6,
                linestyle="None",
                label="Key Pass",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=tc,
                markersize=6,
                linestyle="None",
                label="Shot",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#FFD700",
                markersize=8,
                linestyle="None",
                label="Goal",
            ),
        ],
        fontsize=5.5,
        facecolor=BG_MID,
        edgecolor="none",
        labelcolor="white",
        loc="upper left",
        markerscale=0.9,
        framealpha=0.80,
    )


# ═══════════════════════════════════════════════════════════════════
#  WATERMARK & PAGE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

CREDIT_MAIN = "Created by Mostafa Saad"
CREDIT_TOOLS = "Data: WhoScored  |  xG: Internal V7 event-context/team-stat model  |  xT: Karun Singh"


def _watermark(fig):
    """
    Bottom credit bar — separator + data sources.
    Also paints the unified visual identity (yellow top + side accent bars)
    so every PNG that goes through here matches the PPDA visual identity.
    """
    # ── Unified visual identity: yellow top + side bars ──
    try:
        _gold = "#facc15"
        # top bar
        _top = fig.add_axes([0.0, 0.985, 1.0, 0.012], zorder=20)
        _top.set_facecolor(_gold)
        _top.set_xticks([])
        _top.set_yticks([])
        for _s in _top.spines.values():
            _s.set_visible(False)
        # left side bar
        _side = fig.add_axes([0.0, 0.04, 0.005, 0.945], zorder=20)
        _side.set_facecolor(_gold)
        _side.set_xticks([])
        _side.set_yticks([])
        for _s in _side.spines.values():
            _s.set_visible(False)
    except Exception:
        pass

    # thin separator line
    fig.add_artist(
        plt.Line2D(
            [0.03, 0.97],
            [0.026, 0.026],
            transform=fig.transFigure,
            color="#7DD3FC",
            lw=0.8,
            alpha=0.85,
        )
    )
    # Unified-identity footer label — centred, faded gold, letter-spaced
    fig.text(
        0.50,
        0.018,
        "M A T C H   A N A L Y S I S   R E P O R T",
        ha="center",
        va="center",
        color="#64748b",
        fontsize=7.5,
        fontweight="bold",
        transform=fig.transFigure,
    )
    # Tool credits — dim, italic, just below
    fig.text(
        0.50,
        0.008,
        CREDIT_TOOLS,
        ha="center",
        va="center",
        color="#475569",
        fontsize=6.5,
        fontstyle="italic",
        transform=fig.transFigure,
    )


def _page_header(
    fig, hn, an, hg, ag, hxg, axg, hform, aform, venue, status, page_num, page_title
):
    from matplotlib.patches import FancyBboxPatch

    fig.patch.set_facecolor(BG_DARK)

    # coloured title bar
    ax = fig.add_axes([0.0, 0.968, 1.0, 0.032])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            0.470,
            1,
            boxstyle="square,pad=0",
            facecolor=C_RED,
            alpha=0.90,
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.530, 0),
            0.470,
            1,
            boxstyle="square,pad=0",
            facecolor=C_BLUE,
            alpha=0.90,
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.415, 0),
            0.170,
            1,
            boxstyle="square,pad=0",
            facecolor=BG_DARK,
            alpha=0.95,
            zorder=1,
        )
    )
    ax.text(
        0.235,
        0.52,
        hn,
        ha="center",
        va="center",
        color=_text_on_color(C_RED),
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.765,
        0.52,
        an,
        ha="center",
        va="center",
        color=_text_on_color(C_BLUE),
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.500,
        0.52,
        f"{hg}  —  {ag}",
        ha="center",
        va="center",
        color="#FFD700",
        fontsize=20,
        fontweight="bold",
    )

    # info bar
    ax2 = fig.add_axes([0.0, 0.952, 1.0, 0.016])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_facecolor("#07090f")
    sl, sc = STATUS_BADGE.get(status, ("■ Full Time", "#64748b"))
    ax2.text(
        0.012,
        0.50,
        f"xG: {hxg}  •  {hform}",
        ha="left",
        va="center",
        color=C_RED,
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.text(
        0.500,
        0.50,
        f"{sl}   •   {venue}   •   {page_title}   •   Page {page_num}/2",
        ha="center",
        va="center",
        color=TEXT_DIM,
        fontsize=8,
    )
    ax2.text(
        0.988,
        0.50,
        f"{aform}  •  xG: {axg}",
        ha="right",
        va="center",
        color=C_BLUE,
        fontsize=8.5,
        fontweight="bold",
    )


# ═══════════════════════════════════════════════════════════════════
#  PANEL HELPERS  (each = one team, one metric, independent)
# ═══════════════════════════════════════════════════════════════════


# ── A: Shot Comparison (full-width dual tiles) ─────────────────────
def _panel_shot_comparison(ax, events, info, xg_data):
    ax.set_facecolor(BG_MID)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hn, an = info["home_name"], info["away_name"]
    hd = xg_data.get(hn, {})
    ad = xg_data.get(an, {})
    metrics = [
        ("Goals", hd.get("goals", 0), ad.get("goals", 0), C_GOLD),
        ("xG", hd.get("xG", 0), ad.get("xG", 0), "#a855f7"),
        ("Shots", hd.get("shots", 0), ad.get("shots", 0), "#94a3b8"),
        ("On Target", hd.get("on_target", 0), ad.get("on_target", 0), C_GREEN),
        ("Blocked", hd.get("blocked", 0), ad.get("blocked", 0), C_GOLD),
        ("Big Ch.", hd.get("big_chances", 0), ad.get("big_chances", 0), "#f43f5e"),
        ("Off Target", hd.get("missed", 0), ad.get("missed", 0), "#64748b"),
    ]
    n = len(metrics)
    cw = 1.0 / n

    ax.text(
        0.50,
        0.975,
        "SHOT COMPARISON",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.975,
        hn[:14],
        ha="left",
        va="top",
        color=C_RED,
        fontsize=8.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        0.975,
        an[:14],
        ha="right",
        va="top",
        color=C_BLUE,
        fontsize=8.5,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.plot([0.01, 0.99], [0.91, 0.91], color="#1f2937", lw=0.8, transform=ax.transAxes)

    for i, (lbl, hv, av, col) in enumerate(metrics):
        cx = (i + 0.5) * cw
        tot = (float(hv) + float(av)) or 1
        hr = float(hv) / tot

        ax.add_patch(
            mpatches.FancyBboxPatch(
                (i * cw + 0.006, 0.05),
                cw - 0.012,
                0.84,
                boxstyle="round,pad=0.006",
                transform=ax.transAxes,
                facecolor="#050505",
                edgecolor=col,
                lw=1.3,
                alpha=0.90,
            )
        )

        BAR_Y, BAR_H = 0.35, 0.18
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (i * cw + 0.009, BAR_Y),
                (cw - 0.018) * hr,
                BAR_H,
                boxstyle="round,pad=0.002",
                transform=ax.transAxes,
                facecolor=C_RED,
                alpha=0.88,
            )
        )
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (i * cw + 0.009 + (cw - 0.018) * hr, BAR_Y),
                (cw - 0.018) * (1 - hr),
                BAR_H,
                boxstyle="round,pad=0.002",
                transform=ax.transAxes,
                facecolor=C_BLUE,
                alpha=0.88,
            )
        )

        hv_str = f"{hv:.2f}" if isinstance(hv, float) else str(hv)
        av_str = f"{av:.2f}" if isinstance(av, float) else str(av)
        # Home
        ax.text(
            cx - cw * 0.22,
            0.66,
            hv_str,
            ha="center",
            va="center",
            color=C_RED,
            fontsize=15,
            fontweight="bold",
            transform=ax.transAxes,
            path_effects=[pe.withStroke(linewidth=2, foreground="#000")],
        )
        # Away
        ax.text(
            cx + cw * 0.22,
            0.66,
            av_str,
            ha="center",
            va="center",
            color=C_BLUE,
            fontsize=15,
            fontweight="bold",
            transform=ax.transAxes,
            path_effects=[pe.withStroke(linewidth=2, foreground="#000")],
        )

        ax.text(
            cx,
            0.66,
            "–",
            ha="center",
            va="center",
            color="#475569",
            fontsize=11,
            transform=ax.transAxes,
        )

        ax.text(
            cx,
            0.19,
            lbl,
            ha="center",
            va="center",
            color=col,
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
        )


# ── B: Danger Creation (single team) ──────────────────────────────
def _panel_danger(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Danger Creation — {name}", tc)
    dng = events[
        ((events["is_shot"] == True) | (events["is_key_pass"] == True))
        & (events["team_id"] == tid)
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if dng.empty:
        return
    shots = dng[dng["is_shot"] == True]
    kp = dng[dng["is_key_pass"] == True]
    goals = shots[shots["is_goal"] == True] if not shots.empty else shots
    if not dng.empty:
        ax.scatter(
            dng["x"], dng["y"], c=tc, s=120, alpha=0.07, zorder=2, edgecolors="none"
        )
    if not kp.empty:
        ax.scatter(
            kp["x"],
            kp["y"],
            c="#facc15",
            s=22,
            marker="^",
            alpha=0.78,
            zorder=4,
            edgecolors="none",
        )
    if not shots.empty:
        ax.scatter(
            shots["x"],
            shots["y"],
            c=tc,
            s=30,
            alpha=0.78,
            zorder=4,
            edgecolors="white",
            lw=0.5,
        )
    if not goals.empty:
        ax.scatter(
            goals["x"],
            goals["y"],
            c="#FFD700",
            s=95,
            marker="*",
            zorder=6,
            edgecolors="white",
            lw=0.8,
        )
    ax.text(
        50,
        -4.5,
        f"Shots:{len(shots)}  Key Pass:{len(kp)}  Goals:{len(goals)}",
        ha="center",
        va="top",
        color=tc,
        fontsize=6.5,
        fontweight="bold",
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#facc15",
                markersize=6,
                linestyle="None",
                label="Key Pass",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=tc,
                markersize=6,
                linestyle="None",
                label="Shot",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#FFD700",
                markersize=8,
                linestyle="None",
                label="Goal",
            ),
        ],
        fontsize=5.5,
        facecolor=BG_MID,
        edgecolor="none",
        labelcolor="white",
        loc="upper left",
        markerscale=0.9,
        framealpha=0.80,
    )


# ── C: Goals & Assists table ───────────────────────────────────────
def _panel_goals_table(ax, events, info):
    ax.set_facecolor(BG_MID)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _lbl(ax, "Goals & Assists", TEXT_BRIGHT)
    gdf = events[events["is_goal"] == True].copy()
    if gdf.empty:
        ax.text(
            0.50,
            0.50,
            "No goals recorded",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=10,
            style="italic",
            transform=ax.transAxes,
        )
        return
    rows = []
    for _, r in gdf.sort_values("minute").iterrows():
        is_og = bool(r.get("is_own_goal", False))
        ben_id = r.get("scoring_team", r["team_id"])
        col = OG_COLOR if is_og else (C_RED if ben_id == info["home_id"] else C_BLUE)
        from match_report import (
            classify_goal_type as _classify_goal_type,
            goal_body_part_label as _goal_body_part_label,
        )

        if is_og:
            gtype = "OG"
        else:
            cat, sub = _classify_goal_type(r, events)
            body = _goal_body_part_label(r)
            gtype = (
                f"{cat} - {body}"
                if cat != "Set Piece"
                else f"Set Piece - {sub} - {body}"
            )
        rows.append(
            (r["minute"], r["player"], r["assist_player"], gtype, r["xG"], col, is_og)
        )
    row_h = min(0.78 / max(len(rows), 1), 0.14)
    y = 0.87
    for x_, lbl_, ha_ in [
        (0.07, "MIN", "center"),
        (0.35, "SCORER", "left"),
        (0.62, "ASSIST", "left"),
        (0.82, "TYPE", "center"),
        (0.95, "xG", "right"),
    ]:
        ax.text(
            x_,
            y,
            lbl_,
            ha=ha_,
            va="center",
            color=TEXT_DIM,
            fontsize=7.5,
            fontweight="bold",
            transform=ax.transAxes,
        )
    y -= 0.025
    ax.plot([0.01, 0.99], [y, y], color=GRID_COL, lw=0.8, transform=ax.transAxes)
    y -= 0.008
    for min_, scorer, assist, gtype, xg, col, is_og in rows:
        bg = "#1e0a2e" if is_og else ("#1a0a0a" if col == C_RED else "#090909")
        ax.add_patch(
            plt.Rectangle(
                (0.01, y - row_h * 0.85),
                0.98,
                row_h * 0.82,
                facecolor=bg,
                edgecolor=col,
                lw=0.6,
                alpha=0.9,
                transform=ax.transAxes,
            )
        )
        cy = y - row_h * 0.42
        ax.text(
            0.07,
            cy,
            f"{int(min_)}'",
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.22", facecolor=col, alpha=0.88, edgecolor="none"
            ),
        )
        ax.text(
            0.35,
            cy,
            _short(str(scorer)) if scorer else "—",
            ha="left",
            va="center",
            color=col,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.62,
            cy,
            _short(str(assist)) if assist else "—",
            ha="left",
            va="center",
            color=TEXT_DIM,
            fontsize=8.5,
            transform=ax.transAxes,
        )
        ax.text(
            0.82,
            cy,
            gtype,
            ha="center",
            va="center",
            color=col,
            fontsize=9,
            transform=ax.transAxes,
        )
        ax.text(
            0.97,
            cy,
            f"{xg:.2f}" if xg else "—",
            ha="right",
            va="center",
            color="#facc15",
            fontsize=8.5,
            transform=ax.transAxes,
        )
        y -= row_h


# ── D: Mini Shot Map (single team) ────────────────────────────────
def _panel_shot_mini(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Shot Map — {name}", tc)
    shots = events[
        (events["is_shot"] == True)
        & (events["team_id"] == tid)
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if shots.empty:
        return
    raw_col = (
        "shot_whoscored_type"
        if "shot_whoscored_type" in shots.columns
        else "shot_category"
    )
    for raw_type, (marker, fc, ec, sz, z, _) in SHOT_STYLE_RAW.items():
        sub = shots[shots[raw_col] == raw_type]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            c=fc,
            s=sz * 0.15,
            marker=marker,
            edgecolors=ec,
            linewidths=0.8,
            alpha=0.92,
            zorder=z,
        )
    shot_counts = get_shot_counts(shots)
    n_tot = shot_counts["shots"]
    n_sot = shot_counts["on_target"]
    xg_t = round(float(shots["xG"].sum()), 2)
    ax.text(
        50,
        -4.5,
        f"Shots:{n_tot}  SoT:{n_sot}  xG:{xg_t}",
        ha="center",
        va="top",
        color=tc,
        fontsize=6.5,
        fontweight="bold",
    )


# ── E: xG / xGoT tiles (full-width summary) ───────────────────────
def _panel_xg_tiles(ax, events, info, xg_data):
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hn, an = info["home_name"], info["away_name"]
    hd = xg_data.get(hn, {})
    ad = xg_data.get(an, {})
    hs = events[(events["is_shot"] == True) & (events["team_id"] == info["home_id"])]
    as_ = events[(events["is_shot"] == True) & (events["team_id"] == info["away_id"])]
    hxgot = round(
        hs[
            (
                hs.get("shot_whoscored_type", hs["shot_category"]).isin(
                    ["Goal", "SavedShot", "On Target"]
                )
            )
        ]["xG"].sum(),
        2,
    )
    axgot = round(
        as_[
            (
                as_.get("shot_whoscored_type", as_["shot_category"]).isin(
                    ["Goal", "SavedShot", "On Target"]
                )
            )
        ]["xG"].sum(),
        2,
    )
    tiles = [
        (hn[:12], hd.get("xG", 0), C_RED, "xG"),
        (hn[:12], hxgot, "#F5C542", "xGoT"),
        (hn[:12], hd.get("on_target", 0), C_GREEN, "On Target"),
        (an[:12], axgot, "#F5C542", "xGoT"),
        (an[:12], ad.get("xG", 0), C_BLUE, "xG"),
        (an[:12], ad.get("on_target", 0), C_GREEN, "On Target"),
    ]
    tw = 1.0 / len(tiles)
    for i, (team, val, col, lbl) in enumerate(tiles):
        cx = (i + 0.5) * tw
        ax.add_patch(
            plt.Rectangle(
                (i * tw + 0.003, 0.04),
                tw - 0.006,
                0.92,
                facecolor="#0d1117",
                edgecolor=col,
                lw=1.2,
                alpha=0.9,
                transform=ax.transAxes,
            )
        )
        ax.text(
            cx,
            0.72,
            str(round(val, 2) if isinstance(val, float) else val),
            ha="center",
            va="center",
            color=col,
            fontsize=16,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            cx,
            0.40,
            lbl,
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=9,
            transform=ax.transAxes,
        )
        ax.text(
            cx,
            0.18,
            team,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            transform=ax.transAxes,
        )


# Shot Summary Tiles panel removed by request.
# The old tile visual was not reliable enough because it rebuilt shot totals from raw event buckets
# instead of relying only on official WhoScored/Opta totals.


# ── F: Pass Map with thirds (single team) ─────────────────────────
def _panel_pass_thirds(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Pass Map — {name}", tc)
    p = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if p.empty:
        return
    zone_cfg = [
        ((0, 33), "#64748b", "Own 3rd"),
        ((33, 66), C_GOLD, "Mid 3rd"),
        ((66, 100), C_GREEN, "Final 3rd"),
    ]
    for (x0, x1), col, zlbl in zone_cfg:
        sub = p[(p["x"] >= x0) & (p["x"] < x1)]
        succ = sub[sub["outcome"] == "Successful"]
        fail = sub[sub["outcome"] != "Successful"]
        for df_, alpha, lw in [(succ, 0.55, 0.9), (fail, 0.20, 0.5)]:
            for _, r in df_.iterrows():
                ax.annotate(
                    "",
                    xy=(r["end_x"], r["end_y"]),
                    xytext=(r["x"], r["y"]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=col,
                        lw=lw,
                        alpha=alpha,
                        mutation_scale=4,
                    ),
                    zorder=3,
                )
    ax.axvline(33, color="#475569", lw=0.8, ls=":", alpha=0.6, zorder=5)
    ax.axvline(66, color="#475569", lw=0.8, ls=":", alpha=0.6, zorder=5)
    tot = len(p)
    for (x0, x1), col, zlbl in zone_cfg:
        sub = p[(p["x"] >= x0) & (p["x"] < x1)]
        suc = int((sub["outcome"] == "Successful").sum())
        pct = round(len(sub) / tot * 100) if tot else 0
        cx = (x0 + x1) / 2

        ax.text(
            cx,
            -3.5,
            zlbl,
            ha="center",
            va="top",
            color=col,
            fontsize=7,
            fontweight="bold",
        )

        ax.text(
            cx,
            -8.5,
            f"{len(sub)} passes ({pct}%)",
            ha="center",
            va="top",
            color="#cbd5e1",
            fontsize=6.5,
        )

        ax.text(
            cx,
            -13.0,
            f"{suc} ✓ completed",
            ha="center",
            va="top",
            color=C_GREEN,
            fontsize=6.5,
        )


# ── G: Progressive Passes (single team) ───────────────────────────
def _panel_progressive(ax, events, tid, tc, name):
    _mini_pitch(ax)
    p = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    prog = p[(p["end_x"] - p["x"]) >= 10].copy() if not p.empty else p
    n = len(prog)
    _lbl(ax, f"{name}: {n} Progressive Passes", tc)
    if prog.empty:
        return

    def zc(x):
        if x < 33:
            return "#64748b"
        if x < 66:
            return C_GOLD
        return C_GREEN

    for _, r in prog.iterrows():
        ax.annotate(
            "",
            xy=(r["end_x"], r["end_y"]),
            xytext=(r["x"], r["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color=zc(r["x"]), lw=0.9, alpha=0.55, mutation_scale=5
            ),
            zorder=3,
        )
    n1 = int((prog["x"] < 33).sum())
    n2 = int(((prog["x"] >= 33) & (prog["x"] < 66)).sum())
    n3 = int((prog["x"] >= 66).sum())
    for i, (lv, cl) in enumerate(
        [
            (f"Own 3rd  {n1}({int(n1/n*100) if n else 0}%)", "#64748b"),
            (f"Mid 3rd  {n2}({int(n2/n*100) if n else 0}%)", C_GOLD),
            (f"Final 3rd {n3}({int(n3/n*100) if n else 0}%)", C_GREEN),
        ]
    ):
        ax.text(
            99,
            99 - i * 9,
            lv,
            ha="right",
            va="top",
            color=cl,
            fontsize=6.5,
            fontweight="bold",
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=BG_DARK,
                alpha=0.80,
                edgecolor="none",
            ),
        )


# ── H: Defensive Heatmap (single team) ────────────────────────────
DEFENSIVE_TYPES = {
    "Tackle": (C_RED, "Tackle"),
    "Interception": (C_BLUE, "Int."),
    "BallRecovery": ("#22c55e", "Recovery"),
    "Clearance": (C_GOLD, "Clear."),
    "BlockedShot": (C_GREEN, "Block"),
    "Aerial": ("#a855f7", "Aerial"),
    "Challenge": ("#f97316", "Challenge"),
}


def _blocked_shots_for_team(events, info, team_id) -> int:
    """WhoScored stores BlockedShot on the shooting team; count it for the defence."""
    if "team_id" not in events.columns:
        return 0
    hid = info.get("home_id")
    aid = info.get("away_id")

    def _blocked_shots_by(shooter_id):
        sub = events[events["team_id"] == shooter_id].copy()
        if sub.empty:
            return sub
        hit = pd.Series(False, index=sub.index)
        for col in ("type", "shot_whoscored_type", "shot_category"):
            if col in sub.columns:
                vals = (
                    sub[col]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.replace(r"[^a-z]", "", regex=True)
                )
                hit = hit | vals.isin({"blockedshot", "blocked"})
        if "qualifier_names" in sub.columns:
            hit = hit | sub["qualifier_names"].fillna("").astype(str).str.contains(
                r"\bBlocked\b", case=False, regex=True
            )
        if "is_shot" in sub.columns:
            shot_mask = (sub["is_shot"] == True) | hit
            hit = hit & shot_mask
        return sub[hit].copy()

    direct = len(_blocked_shots_by(team_id))
    opp_id = aid if team_id == hid else (hid if team_id == aid else None)
    if opp_id is None:
        return direct
    opp_blocked = len(_blocked_shots_by(opp_id))
    if not opp_blocked:
        opp_side = "away" if team_id == hid else "home"
        mc = (info.get("matchcentre_stats", {}) or {}).get(opp_side, {}) or {}
        raw_blocked = mc.get("blocked", 0)
        try:
            opp_blocked = int(float(raw_blocked or 0))
        except (TypeError, ValueError):
            opp_blocked = 0
    return opp_blocked if opp_blocked else direct


def _defensive_events_for_team(events, info, team_id):
    """Defensive events plus opponent blocked shots mapped onto this team."""
    if "type" not in events.columns or "team_id" not in events.columns:
        return pd.DataFrame()
    own_types = [t for t in DEFENSIVE_TYPES.keys() if t != "BlockedShot"]
    own = events[(events["team_id"] == team_id) & events["type"].isin(own_types)].copy()
    hid = info.get("home_id")
    aid = info.get("away_id")
    opp_id = aid if team_id == hid else (hid if team_id == aid else None)

    def _blocked_shots_by(shooter_id):
        sub = events[events["team_id"] == shooter_id].copy()
        if sub.empty:
            return sub
        hit = pd.Series(False, index=sub.index)
        for col in ("type", "shot_whoscored_type", "shot_category"):
            if col in sub.columns:
                vals = (
                    sub[col]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.replace(r"[^a-z]", "", regex=True)
                )
                hit = hit | vals.isin({"blockedshot", "blocked"})
        if "qualifier_names" in sub.columns:
            hit = hit | sub["qualifier_names"].fillna("").astype(str).str.contains(
                r"\bBlocked\b", case=False, regex=True
            )
        if "is_shot" in sub.columns:
            shot_mask = (sub["is_shot"] == True) | hit
            hit = hit & shot_mask
        return sub[hit].copy()

    if opp_id is not None:
        blocks = _blocked_shots_by(opp_id)
        if blocks.empty:
            blocks = _blocked_shots_by(team_id)
    else:
        blocks = _blocked_shots_by(team_id)
    if not blocks.empty:
        blocks["team_id"] = team_id
        blocks["type"] = "BlockedShot"
    if own.empty:
        return blocks
    if blocks.empty:
        return own
    return pd.concat([own, blocks], ignore_index=True, sort=False)


def _panel_defensive_heatmap(ax, events, tid, tc, name, info=None):
    """
    Defensive Actions Heatmap — single team.
    Each type = unique colour + marker. Legend outside pitch (right side).
    """
    DEF_STYLE = {
        "Tackle": ("#f43f5e", "D", "Tackle"),
        "Interception": ("#3b82f6", "^", "Interception"),
        "BallRecovery": ("#22c55e", "o", "Ball Recovery"),
        "Clearance": ("#f59e0b", "s", "Clearance"),
        "BlockedShot": ("#a855f7", "P", "Blocked Shot"),
        "Aerial": ("#06b6d4", "*", "Aerial"),
        "Challenge": ("#fb923c", "v", "Challenge"),
        "Foul": ("#94a3b8", "x", "Foul"),
    }

    # extend xlim to give legend room on the right
    _mini_pitch(ax, bg="#040d04")
    ax.set_xlim(-3, 130)  # override default → extra 27 units for legend
    ax.set_ylim(-10, 113)

    # title — use transAxes so extended xlim doesn't affect it
    ax.text(
        0.38,
        1.025,
        f"Defensive Actions — {name}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        color=tc,
        fontsize=8.5,
        fontweight="bold",
        zorder=20,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#0d1117",
            edgecolor=tc,
            linewidth=1.0,
            alpha=0.92,
        ),
    )

    def_ev = _defensive_events_for_team(events, info or {}, tid)
    if "x" in def_ev.columns and "y" in def_ev.columns:
        def_ev = def_ev[def_ev[["x", "y"]].notna().all(axis=1)].copy()

    if def_ev.empty:
        ax.text(
            50,
            50,
            "No defensive data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return

    # ── KDE density background ───────────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap as LSC

    hm_cmap = LSC.from_list(
        "dhm",
        ["#000000", "#050505", "#0A0A0A", "#0f4c2a", "#a16207", "#dc2626", "#ff0a47"],
        N=256,
    )
    xs, ys = def_ev["x"].values, def_ev["y"].values
    if len(xs) >= 6:
        try:
            from scipy.stats import gaussian_kde

            gx, gy = np.mgrid[0:100:50j, 0:100:50j]
            kde = gaussian_kde(np.vstack([xs, ys]), bw_method=0.22)
            z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            ax.contourf(gx, gy, z, levels=10, cmap=hm_cmap, alpha=0.55, zorder=2)
        except Exception:
            pass

    # ── Scatter each type ────────────────────────────────────────
    for dtype, (col, mkr, _lbl) in DEF_STYLE.items():
        sub = def_ev[def_ev["type"] == dtype]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            c=col,
            marker=mkr,
            s=40,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.92,
            zorder=5,
        )

    # ── Legend panel (right of pitch: x 104–128) ─────────────────
    counts = def_ev["type"].value_counts()
    lx0 = 104
    ax.add_patch(
        plt.Rectangle(
            (lx0, -2),
            25,
            104,
            facecolor="#080f08",
            alpha=0.85,
            edgecolor="#3A3A3A",
            lw=0.8,
            zorder=6,
        )
    )
    ax.text(
        lx0 + 12.5,
        99,
        "LEGEND",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=6.5,
        fontweight="bold",
        zorder=7,
    )

    ly = 93
    for dtype, (col, mkr, lbl_txt) in DEF_STYLE.items():
        n = int(counts.get(dtype, 0))
        if n == 0:
            continue
        ax.scatter(
            [lx0 + 3],
            [ly],
            c=col,
            marker=mkr,
            s=32,
            edgecolors="white",
            linewidths=0.5,
            zorder=8,
        )
        ax.text(
            lx0 + 6.5,
            ly,
            f"{lbl_txt}",
            ha="left",
            va="center",
            color=col,
            fontsize=6.2,
            fontweight="bold",
            zorder=8,
        )
        ax.text(
            lx0 + 6.5,
            ly - 4.5,
            f"n = {n}",
            ha="left",
            va="center",
            color="#94a3b8",
            fontsize=5.5,
            zorder=8,
        )
        ly -= 13

    # ── Zone footer (below pitch) ─────────────────────────────────
    zone_data = [("Def 3rd", 0, 33), ("Mid 3rd", 33, 66), ("Att 3rd", 66, 100)]
    parts = []
    for zlbl, x0, x1 in zone_data:
        n = int(def_ev[(def_ev["x"] >= x0) & (def_ev["x"] < x1)].shape[0])
        parts.append(f"{zlbl}: {n}")
    ax.text(
        50,
        -7,
        "   ".join(parts),
        ha="center",
        va="top",
        color=tc,
        fontsize=6.5,
        fontweight="bold",
    )


# ── I: Pass Network (single team, mini) ───────────────────────────
def _panel_pass_network(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Pass Network — {name}", tc)
    p = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events["player_id"].notna()
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if p.empty:
        return
    avg = p.groupby(["player_id", "player"], dropna=True)[["x", "y"]].mean()
    cnt = p.groupby(["player_id", "player"], dropna=True).size().rename("n")
    roles = (
        p.groupby(["player_id", "player"], dropna=True)["player_role"].first()
        if "player_role" in p.columns
        else None
    )
    avg = avg.join(cnt)
    if roles is not None:
        avg = avg.join(roles.rename("role"))
    mx_n = avg["n"].max() if not avg.empty else 1
    for key, row in avg.iterrows():
        pl = key[1] if isinstance(key, tuple) else key
        s = 18 + row["n"] / mx_n * 90
        role = row.get("role", "") if hasattr(row, "get") else ""
        c = tc
        if role == "sub_in":
            c = COLOR_SUB_IN
        elif role == "sub_out":
            c = COLOR_SUB_OUT
        elif role == "both_sub":
            c = COLOR_BOTH_SUB
        elif role == "red_card":
            c = COLOR_RED_CARD
        badge = {"sub_in": "↑", "sub_out": "↓", "both_sub": "↕", "red_card": "RC"}.get(
            role, ""
        )
        ax.scatter(row["x"], row["y"], c=c, s=s, edgecolors="white", lw=0.7, zorder=5)
        nm = pl.split()[-1][:8] if pl else ""
        if badge:
            nm = f"{nm} {badge}"
        ax.text(
            row["x"],
            row["y"] + 3.5,
            nm,
            ha="center",
            va="bottom",
            color="white",
            fontsize=5.5,
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.14",
                facecolor="#000",
                alpha=0.68,
                edgecolor="none",
            ),
        )


# ── J: Avg Positions (single team) ────────────────────────────────
def _panel_avg_position(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax, f"Avg Positions — {name}", tc)
    ev = events[
        (events["team_id"] == tid)
        & events["player_id"].notna()
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if ev.empty:
        return
    avg = ev.groupby(["player_id", "player"], dropna=True)[["x", "y"]].mean()
    cnt = ev.groupby(["player_id", "player"], dropna=True).size()
    roles = (
        ev.groupby(["player_id", "player"], dropna=True)["player_role"].first()
        if "player_role" in ev.columns
        else None
    )
    mx = cnt.max() if not cnt.empty else 1
    for key, row in avg.iterrows():
        pl = key[1] if isinstance(key, tuple) else key
        n = cnt.get(key, 1)
        role = roles.get(key, "") if roles is not None else ""
        c = tc
        if role == "sub_in":
            c = COLOR_SUB_IN
        elif role == "sub_out":
            c = COLOR_SUB_OUT
        elif role == "both_sub":
            c = COLOR_BOTH_SUB
        elif role == "red_card":
            c = COLOR_RED_CARD
        badge = {"sub_in": "↑", "sub_out": "↓", "both_sub": "↕", "red_card": "RC"}.get(
            role, ""
        )
        ax.scatter(
            row["x"],
            row["y"],
            c=c,
            s=28 + n / mx * 82,
            edgecolors="white",
            lw=0.7,
            alpha=0.88,
            zorder=4,
        )
        nm = pl.split()[-1][:7] if pl else ""
        if badge:
            nm = f"{nm} {badge}"
        ax.text(
            row["x"],
            row["y"] + 3.2,
            nm,
            ha="center",
            va="bottom",
            color="white",
            fontsize=5.5,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="#000",
                alpha=0.68,
                edgecolor="none",
            ),
        )


# ── K: Match Statistics table ──────────────────────────────────────
def _panel_match_stats(ax, events, info, xg_data):
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]
    hd = xg_data.get(hn, {})
    ad = xg_data.get(an, {})
    hp = int(events[(events["is_pass"] == True) & (events["team_id"] == hid)].shape[0])
    ap = int(events[(events["is_pass"] == True) & (events["team_id"] == aid)].shape[0])
    tot_p = (hp + ap) or 1
    hps = int(
        events[
            (events["is_pass"] == True)
            & (events["team_id"] == hid)
            & (events["outcome"] == "Successful")
        ].shape[0]
    )
    aps = int(
        events[
            (events["is_pass"] == True)
            & (events["team_id"] == aid)
            & (events["outcome"] == "Successful")
        ].shape[0]
    )
    hkp = int(
        events[(events["is_key_pass"] == True) & (events["team_id"] == hid)].shape[0]
    )
    akp = int(
        events[(events["is_key_pass"] == True) & (events["team_id"] == aid)].shape[0]
    )
    hxt = (
        round(events[(events["team_id"] == hid) & events["xT"].notna()]["xT"].sum(), 2)
        if "xT" in events.columns
        else 0
    )
    axt = (
        round(events[(events["team_id"] == aid) & events["xT"].notna()]["xT"].sum(), 2)
        if "xT" in events.columns
        else 0
    )
    stats = [
        (
            "Possession",
            f"{round(hp/tot_p*100,1)}%",
            f"{round(ap/tot_p*100,1)}%",
            hp / tot_p,
            C_GOLD,
        ),
        (
            "Passes (Acc)",
            f"{hp}({hps})",
            f"{ap}({aps})",
            hp / ((hp + ap) or 1),
            "#94a3b8",
        ),
        (
            "Shots (SoT)",
            f"{hd.get('shots',0)}({hd.get('on_target',0)})",
            f"{ad.get('shots',0)}({ad.get('on_target',0)})",
            hd.get("shots", 0) / ((hd.get("shots", 0) + ad.get("shots", 0)) or 1),
            "#64748b",
        ),
        (
            "xG",
            str(hd.get("xG", 0)),
            str(ad.get("xG", 0)),
            hd.get("xG", 0) / ((hd.get("xG", 0) + ad.get("xG", 0)) or 1),
            "#a855f7",
        ),
        ("xT", str(hxt), str(axt), hxt / ((hxt + axt) or 1), "#22c55e"),
        ("Key Passes", str(hkp), str(akp), hkp / ((hkp + akp) or 1), "#F5C542"),
        (
            "Big Chances",
            str(hd.get("big_chances", 0)),
            str(ad.get("big_chances", 0)),
            hd.get("big_chances", 0)
            / ((hd.get("big_chances", 0) + ad.get("big_chances", 0)) or 1),
            "#f43f5e",
        ),
    ]
    ax.text(
        0.50,
        0.99,
        "MATCH STATISTICS",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=9.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.10,
        0.94,
        hn[:13],
        ha="left",
        va="top",
        color=C_RED,
        fontsize=8,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.90,
        0.94,
        an[:13],
        ha="right",
        va="top",
        color=C_BLUE,
        fontsize=8,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.plot([0.02, 0.98], [0.90, 0.90], color=GRID_COL, lw=0.8, transform=ax.transAxes)
    y = 0.85
    step = 0.118
    for lbl_, hv, av, hr, col in stats:
        # Full-width strip background
        strip_x, strip_y, strip_h = 0.02, y - 0.050, 0.052
        strip_w = 0.96
        # Home side fill
        ax.add_patch(
            plt.Rectangle(
                (strip_x, strip_y),
                strip_w * hr,
                strip_h,
                facecolor=C_RED,
                alpha=0.32,
                transform=ax.transAxes,
            )
        )
        # Away side fill
        ax.add_patch(
            plt.Rectangle(
                (strip_x + strip_w * hr, strip_y),
                strip_w * (1 - hr),
                strip_h,
                facecolor=C_BLUE,
                alpha=0.32,
                transform=ax.transAxes,
            )
        )
        # Thin outline around full strip
        ax.add_patch(
            plt.Rectangle(
                (strip_x, strip_y),
                strip_w,
                strip_h,
                facecolor="none",
                edgecolor=col,
                lw=0.7,
                alpha=0.55,
                transform=ax.transAxes,
            )
        )
        # Stat label — centred INSIDE the strip
        ax.text(
            0.50,
            y - 0.024,
            lbl_,
            ha="center",
            va="center",
            color="white",
            fontsize=7.5,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#000000")],
        )
        # Home value — left edge inside strip (dynamic contrast on C_RED strip)
        _home_val_col = _text_on_color(C_RED, light="#ffffff", dark="#111827")
        # Ensure readable on the dark BG_DARK behind the strip
        if _relative_luminance(C_RED) < 0.20:
            _home_val_col = "#ffd6d6"  # soft tint on dark bars
        ax.text(
            0.045,
            y - 0.024,
            str(hv),
            ha="left",
            va="center",
            color=_home_val_col,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=5,
            path_effects=[
                pe.withStroke(
                    linewidth=1.5,
                    foreground="#000000" if _home_val_col != "#111827" else "#ffffff",
                )
            ],
        )
        # Away value — right edge inside strip (dynamic contrast on C_BLUE strip)
        _away_val_col = _text_on_color(C_BLUE, light="#ffffff", dark="#111827")
        if _relative_luminance(C_BLUE) < 0.20:
            _away_val_col = "#d6e8ff"  # soft tint on dark bars
        ax.text(
            0.955,
            y - 0.024,
            str(av),
            ha="right",
            va="center",
            color=_away_val_col,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=5,
            path_effects=[
                pe.withStroke(
                    linewidth=1.5,
                    foreground="#000000" if _away_val_col != "#111827" else "#ffffff",
                )
            ],
        )
        y -= step


# ── L: xT per Minute (both teams, full-width) ─────────────────────
def _panel_xt_minute(ax, events, info):
    """
    Diverging bar chart: Home xT bars up (red), Away xT bars down (blue).
    Team names + totals shown in clear top-left / top-right boxes.
    HT and FT markers with labels above the bars.
    """
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]

    ax.set_facecolor(BG_DARK)
    _lbl(ax, "xT per Minute  (▲ Home  |  ▼ Away)", TEXT_BRIGHT)

    if "xT" not in events.columns:
        ax.text(
            0.5,
            0.5,
            "No xT data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            transform=ax.transAxes,
            fontsize=10,
        )
        return

    xt = events[
        events["xT"].notna() & (events["xT"] > 0) & (events["outcome"] == "Successful")
    ].copy()

    hxt = xt[xt["team_id"] == hid].groupby("minute")["xT"].sum()
    axt = xt[xt["team_id"] == aid].groupby("minute")["xT"].sum()
    mins = list(range(1, 96))
    h_vals = [hxt.get(m, 0) for m in mins]
    a_vals = [-axt.get(m, 0) for m in mins]

    ax.bar(mins, h_vals, color=C_RED, alpha=0.72, width=0.85, zorder=3)
    ax.bar(mins, a_vals, color=C_BLUE, alpha=0.72, width=0.85, zorder=3)
    ax.axhline(0, color="#94a3b8", lw=0.9, alpha=0.55, zorder=4)
    # 5-minute rolling average overlay
    import pandas as _pd2

    _hv = _pd2.Series(h_vals).rolling(5, center=True, min_periods=1).mean()
    _av = _pd2.Series(a_vals).rolling(5, center=True, min_periods=1).mean()
    ax.plot(mins, _hv, color=C_RED, lw=2.0, alpha=0.92, zorder=5)
    ax.plot(mins, _av, color=C_BLUE, lw=2.0, alpha=0.92, zorder=5)

    # HT / FT markers
    ymax = max(max(h_vals + [0.001]), abs(min(a_vals + [-0.001])))
    for xp, lb in [(45, "HT"), (90, "FT")]:
        ax.axvline(xp, color=C_GOLD, lw=1.2, ls="--", alpha=0.65, zorder=2)
        ax.text(
            xp + 0.8,
            ymax * 0.90,
            lb,
            color=ACCENT_TEXT,
            fontsize=7.5,
            fontweight="bold",
            va="top",
        )

    # xT totals
    ht_total = round(xt[xt["team_id"] == hid]["xT"].sum(), 3)
    at_total = round(xt[xt["team_id"] == aid]["xT"].sum(), 3)

    for xpos, name, col, ha_, xT_val in [
        (0.01, hn, C_RED, "left", ht_total),
        (0.99, an, C_BLUE, "right", at_total),
    ]:
        ax.text(
            xpos,
            0.97,
            f"{name[:12]}",
            transform=ax.transAxes,
            ha=ha_,
            va="top",
            color=col,
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=BG_MID,
                alpha=0.90,
                edgecolor=col,
                lw=0.8,
            ),
        )
        ax.text(
            xpos,
            0.80,
            f"xT: {xT_val}",
            transform=ax.transAxes,
            ha=ha_,
            va="top",
            color=TEXT_DIM,
            fontsize=8,
        )

    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=8, labelpad=4)
    ax.set_ylabel("xT / min", color=TEXT_DIM, fontsize=8, labelpad=4)
    ax.set_xlim(0, 96)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(GRID_COL)
    ax.grid(axis="y", alpha=0.10, color=GRID_COL)


# ── M: Crosses (both teams) ────────────────────────────────────────
def _panel_crosses(ax, events, info):
    _mini_pitch(ax)
    _lbl(ax, "Crosses", TEXT_DIM)
    if "is_cross" not in events.columns:
        ax.text(
            50,
            50,
            "No cross data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return
    crs = events[
        (events["is_cross"] == True)
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if crs.empty:
        ax.text(
            50,
            50,
            "No crosses recorded",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return
    for tid, col in [(info["home_id"], C_RED), (info["away_id"], C_BLUE)]:
        tc_ = crs[crs["team_id"] == tid]
        for _, r in tc_.iterrows():
            succ = r.get("outcome", "") == "Successful"
            ax.annotate(
                "",
                xy=(r["end_x"], r["end_y"]),
                xytext=(r["x"], r["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=1.3 if succ else 0.6,
                    alpha=0.82 if succ else 0.30,
                    mutation_scale=6,
                ),
                zorder=4,
            )
        eff = int(tc_[tc_["outcome"] == "Successful"].shape[0])
        nm = info["home_name"][:8] if tid == info["home_id"] else info["away_name"][:8]
        xp = 2 if tid == info["home_id"] else 98
        ax.text(
            xp,
            -4.5,
            f"{nm}: {len(tc_)} ({eff} eff.)",
            ha="left" if tid == info["home_id"] else "right",
            va="top",
            color=col,
            fontsize=6.5,
            fontweight="bold",
        )


# ── M2: Crosses — single team ──────────────────────────────────────
def _panel_crosses_team(ax, events, tid, tc, name):
    """
    Crosses for a single team.
    Successful = thick solid arrow, failed = thin faded arrow.
    Cross origin dots shown at pitch position.
    Left / right breakdown in top corners.
    """
    _mini_pitch(ax)
    _lbl(ax, f"Crosses — {name}", tc)

    if "is_cross" not in events.columns:
        ax.text(
            50,
            50,
            "No cross data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return

    crs = events[
        (events["is_cross"] == True)
        & (events["team_id"] == tid)
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()

    if crs.empty:
        ax.text(
            50,
            50,
            "No crosses recorded",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return

    succ_crs = crs[crs["outcome"] == "Successful"]
    fail_crs = crs[crs["outcome"] != "Successful"]
    n_succ = len(succ_crs)
    n_fail = len(fail_crs)
    n_total = len(crs)

    # ── Draw arrows ─────────────────────────────────────────────
    for _, r in fail_crs.iterrows():
        ax.annotate(
            "",
            xy=(r["end_x"], r["end_y"]),
            xytext=(r["x"], r["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color=tc, lw=0.6, alpha=0.28, mutation_scale=5
            ),
            zorder=3,
        )

    for _, r in succ_crs.iterrows():
        # Glow trail
        ax.annotate(
            "",
            xy=(r["end_x"], r["end_y"]),
            xytext=(r["x"], r["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color="white", lw=4.0, alpha=0.08, mutation_scale=10
            ),
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(r["end_x"], r["end_y"]),
            xytext=(r["x"], r["y"]),
            arrowprops=dict(
                arrowstyle="-|>", color=tc, lw=1.6, alpha=0.88, mutation_scale=8
            ),
            zorder=5,
        )

    # ── Origin dots ─────────────────────────────────────────────
    ax.scatter(
        crs["x"], crs["y"], c=tc, s=22, alpha=0.72, edgecolors="white", lw=0.5, zorder=6
    )

    # ── Left / Right breakdown ───────────────────────────────────
    left_n = int((crs["y"] < 40).sum())
    right_n = int((crs["y"] > 60).sum())
    centre_n = n_total - left_n - right_n

    for xp, yp, ha_, txt, col_ in [
        (2, 98, "left", f"Left: {left_n}", tc),
        (50, 98, "center", f"Ctr: {centre_n}", TEXT_DIM),
        (98, 98, "right", f"Right: {right_n}", tc),
    ]:
        ax.text(
            xp,
            yp,
            txt,
            ha=ha_,
            va="top",
            color=col_,
            fontsize=6.8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=BG_DARK,
                alpha=0.78,
                edgecolor="none",
            ),
            zorder=7,
        )

    # ── Summary footer ───────────────────────────────────────────
    acc_pct = round(n_succ / n_total * 100) if n_total else 0
    ax.text(
        50,
        -5.0,
        f"Total: {n_total}  ●  Effective: {n_succ} ({acc_pct}%)  ●  Missed: {n_fail}",
        ha="center",
        va="top",
        color=tc,
        fontsize=7,
        fontweight="bold",
    )

    # ── Legend ───────────────────────────────────────────────────
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=tc, lw=1.8, alpha=0.92, label=f"Effective ({n_succ})"
            ),
            Line2D([0], [0], color=tc, lw=0.6, alpha=0.30, label=f"Missed ({n_fail})"),
        ],
        fontsize=6.5,
        facecolor=BG_MID,
        edgecolor="none",
        labelcolor="white",
        loc="upper right",
        framealpha=0.85,
        markerscale=0.9,
    )


# ── N: Territorial Control bar ────────────────────────────────────
def _panel_territorial(ax, events, info):
    """
    Horizontal stacked bar per zone.
    Home (red) = left portion; Away (blue) = right portion.
    Team names at top; percentages and counts labelled on bars.
    """
    ax.set_facecolor(BG_MID)
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]

    # ── Team name headers ─────────────────────────────────────────
    ax.text(
        0.20,
        1.08,
        hn[:14],
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        color=C_RED,
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.80,
        1.08,
        an[:14],
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        color=C_BLUE,
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.50,
        1.08,
        "Territorial Control",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        color=TEXT_DIM,
        fontsize=8,
        fontstyle="italic",
    )

    zones = [
        ("Own Third", (0, 33)),
        ("Mid Third", (33, 66)),
        ("Final Third", (66, 100)),
    ]
    ev = events[events["x"].notna()].copy()

    for i, (zlbl, (x0, x1)) in enumerate(zones):
        hev = ev[(ev["team_id"] == hid) & (ev["x"] >= x0) & (ev["x"] < x1)]
        aev = ev[(ev["team_id"] == aid) & (ev["x"] >= x0) & (ev["x"] < x1)]
        h_n = len(hev)
        a_n = len(aev)
        tot = (h_n + a_n) or 1
        hr = h_n / tot

        # bars — with slight glow edge
        ax.barh(
            i,
            hr,
            height=0.64,
            color=C_RED,
            alpha=0.85,
            left=0,
            edgecolor="#ff6b7a",
            linewidth=0.6,
        )
        ax.barh(
            i,
            1 - hr,
            height=0.64,
            color=C_BLUE,
            alpha=0.85,
            left=hr,
            edgecolor="#5ba3ff",
            linewidth=0.6,
        )

        # % inside bar — dynamic text contrast
        if hr > 0.08:
            _h_txt = _text_on_color(C_RED)
            _h_stroke = "#000000" if _h_txt == "#ffffff" else "#ffffff"
            ax.text(
                hr / 2,
                i,
                f"{h_n}  ({hr*100:.0f}%)",
                ha="center",
                va="center",
                color=_h_txt,
                fontsize=8.5,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.0, foreground=_h_stroke)],
            )
        if (1 - hr) > 0.08:
            _a_txt = _text_on_color(C_BLUE)
            _a_stroke = "#000000" if _a_txt == "#ffffff" else "#ffffff"
            ax.text(
                hr + (1 - hr) / 2,
                i,
                f"({(1-hr)*100:.0f}%)  {a_n}",
                ha="center",
                va="center",
                color=_a_txt,
                fontsize=8.5,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.0, foreground=_a_stroke)],
            )

    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels(
        [z[0] for z in zones], color=TEXT_BRIGHT, fontsize=9, fontweight="bold"
    )
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.tick_params(left=False)
    ax.set_ylim(-0.5, len(zones) - 0.4)
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_visible(False)
    ax.set_facecolor(BG_MID)


# ── O: Possession Donut (single team) ─────────────────────────────
def _panel_possession_donut(ax, events, tid, tc, name):
    ax.set_facecolor(BG_MID)
    _lbl(ax, f"Ball Touches — {name}", tc)
    ev = events[events[["x", "y"]].notna().all(axis=1)].copy()
    total = len(ev)
    team = int((ev["team_id"] == tid).sum())
    pct = round(team / total * 100, 1) if total else 0
    ax.pie(
        [pct, 100 - pct],
        colors=[tc, "#0A0A0A"],
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor=BG_DARK, lw=1.5),
    )
    ax.text(
        0,
        0,
        f"{pct}%",
        ha="center",
        va="center",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)


# ── P: GK Saves ───────────────────────────────────────────────────
def _panel_gk_saves(ax, events, info):
    """
    Goalkeeper Saves — two goal frames side by side.
    Saves plotted at actual lateral position; height from end_x or random fallback.
    Colour = xG intensity (green→yellow→red).
    Shape  = shot type (circle/triangle/diamond/star/pentagon).
    """
    import matplotlib.colors as mcolors

    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-14, 52)
    ax.axis("off")

    # title
    ax.text(
        50,
        50,
        "Goalkeeper Saves",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=9,
        fontweight="bold",
    )

    GW, GH = 36, 22  # goal width / height in axes units

    # (frame_cx, frame_by, gk_team_id, shoot_team_id, colour, gk_label)
    FRAMES = [
        (26, 10, info["home_id"], info["away_id"], C_RED, info["home_name"]),
        (74, 10, info["away_id"], info["home_id"], C_BLUE, info["away_name"]),
    ]

    SHOT_MKR = {
        "VolleyShot": ("D", 50, "Volley"),
        "Header": ("^", 50, "Header"),
        "FreekickShot": ("p", 50, "Free Kick"),
        "PenaltyShot": ("*", 70, "Penalty"),
    }
    DEFAULT_MKR = ("o", 44, "Open Play")

    xg_cmap = mcolors.LinearSegmentedColormap.from_list(
        "xg_sv", ["#22c55e", "#facc15", "#ef4444"], N=256
    )

    for cx, by, gk_tid, shoot_tid, col, gk_name in FRAMES:
        gx0, gx1 = cx - GW / 2, cx + GW / 2
        gy0, gy1 = by, by + GH

        # ── Goal frame ───────────────────────────────────────────
        ax.plot(
            [gx0, gx1, gx1, gx0, gx0],
            [gy0, gy0, gy1, gy1, gy0],
            color="white",
            lw=2.8,
            alpha=0.95,
            zorder=4,
            solid_capstyle="round",
        )

        # Net grid
        for nx in np.linspace(gx0 + 2.5, gx1 - 2.5, 7):
            ax.plot(
                [nx, nx], [gy0, gy1], color="#374151", lw=0.55, alpha=0.50, zorder=2
            )
        for ny in np.linspace(gy0 + 2.5, gy1 - 2.5, 3):
            ax.plot(
                [gx0, gx1], [ny, ny], color="#374151", lw=0.55, alpha=0.50, zorder=2
            )

        # Zone dividers: thirds (vertical) + half-height (horizontal)
        for xd in [gx0 + GW / 3, gx0 + 2 * GW / 3]:
            ax.plot(
                [xd, xd],
                [gy0, gy1],
                color="#6b7280",
                lw=0.9,
                ls="--",
                alpha=0.60,
                zorder=3,
            )
        ymid = by + GH / 2
        ax.plot(
            [gx0, gx1],
            [ymid, ymid],
            color="#6b7280",
            lw=0.9,
            ls="--",
            alpha=0.60,
            zorder=3,
        )

        # Zone text labels
        for xi, xt in [(gx0 + GW * 0.17, "L"), (cx, "C"), (gx0 + GW * 0.83, "R")]:
            ax.text(
                xi,
                gy0 + 1.5,
                xt,
                ha="center",
                va="bottom",
                color="#9ca3af",
                fontsize=6,
                zorder=5,
            )
        ax.text(
            gx0 - 1.0,
            ymid + GH * 0.25,
            "High",
            ha="right",
            va="center",
            color="#9ca3af",
            fontsize=5.5,
            rotation=90,
            zorder=5,
        )
        ax.text(
            gx0 - 1.0,
            ymid - GH * 0.25,
            "Low",
            ha="right",
            va="center",
            color="#9ca3af",
            fontsize=5.5,
            rotation=90,
            zorder=5,
        )

        # ── All saves by shoot_tid ────────────────────────────────
        saves_all = events[
            (events["team_id"] == shoot_tid) & (events["shot_category"] == "On Target")
        ].copy()
        n_saves = len(saves_all)

        # Team label + save count below frame
        ax.text(
            cx,
            gy0 - 2.5,
            f"{gk_name[:14]}",
            ha="center",
            va="top",
            color=col,
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            cx,
            gy0 - 5.5,
            f"{n_saves} saves",
            ha="center",
            va="top",
            color=TEXT_DIM,
            fontsize=7,
        )

        if saves_all.empty:
            ax.text(
                cx,
                by + GH / 2,
                "No saves",
                ha="center",
                va="center",
                color=TEXT_DIM,
                fontsize=7,
            )
            continue

        np.random.seed(42)
        for _, row in saves_all.iterrows():
            # Lateral (X on frame): use end_y if available else random
            if pd.notna(row.get("end_y")):
                ey = float(row["end_y"])
                # WhoScored end_y: 0–100 across pitch width
                # Goal is between y≈36.8 and y≈63.2  →  remap to full frame width
                ey_norm = (ey - 36.8) / (63.2 - 36.8)
                sx_i = gx0 + np.clip(ey_norm, -0.05, 1.05) * GW
            else:
                sx_i = np.random.uniform(gx0 + 1, gx1 - 1)

            # Height (Y on frame): use goalMouthZ if present, else end_x proxy
            gmz = row.get("goal_mouth_z", None)
            if pd.notna(gmz):
                # goal_mouth_z typically 0–2.44 m
                sy_i = gy0 + (float(gmz) / 2.44) * GH
            elif pd.notna(row.get("end_x")):
                ex = float(row["end_x"])
                # shots heading toward goal: end_x 83.5–100
                # higher = closer to near post (lower height typically)
                sy_i = gy0 + (1 - (ex - 83.5) / 16.5) * GH
            else:
                sy_i = np.random.uniform(gy0 + 1, gy1 - 1)

            sx_i = np.clip(sx_i, gx0 + 0.5, gx1 - 0.5)
            sy_i = np.clip(sy_i, gy0 + 0.5, gy1 - 0.5)

            xg_v = float(row.get("xG") or 0.15)
            rgba = xg_cmap(min(xg_v, 1.0))

            stype = str(row.get("shot_type", ""))
            mkr, sz, _ = SHOT_MKR.get(stype, DEFAULT_MKR)

            ax.scatter(
                [sx_i],
                [sy_i],
                c=[rgba],
                marker=mkr,
                s=sz,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.95,
                zorder=6,
            )
            ax.text(
                sx_i,
                sy_i + 1.5,
                f"{xg_v:.2f}",
                ha="center",
                va="bottom",
                color="#fde68a",
                fontsize=4.5,
                fontweight="bold",
                zorder=7,
            )

    # ── Shared legend at bottom ───────────────────────────────────
    # xG colour scale
    ax.text(
        50,
        -2,
        "Colour = xG intensity:",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=6,
        style="italic",
    )
    for i, (xg_v, lbl_) in enumerate([(0.05, "Low"), (0.30, "Mid"), (0.65, "High")]):
        xi = 30 + i * 14
        rgba = xg_cmap(xg_v)
        ax.scatter(
            [xi],
            [-5.5],
            c=[rgba],
            marker="o",
            s=30,
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
        )
        ax.text(
            xi + 1.8, -5.5, lbl_, ha="left", va="center", color=TEXT_DIM, fontsize=5.5
        )

    ax.text(
        50,
        -7.5,
        "Shape = Shot type:",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=6,
        style="italic",
    )
    shape_items = [
        ("o", "Open Play"),
        ("^", "Header"),
        ("D", "Volley"),
        ("p", "Free Kick"),
        ("*", "Penalty"),
    ]
    for i, (mkr, lbl_) in enumerate(shape_items):
        xi = 12 + i * 17
        ax.scatter(
            [xi],
            [-10.5],
            c="white",
            marker=mkr,
            s=26,
            edgecolors="#6b7280",
            linewidths=0.5,
            zorder=5,
        )
        ax.text(
            xi + 2, -10.5, lbl_, ha="left", va="center", color=TEXT_DIM, fontsize=5.5
        )


# ═══════════════════════════════════════════════════════════════════
#  PAGE 1 — ATTACK REPORT  (no duplicates with Figs 1-10)
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  PAGE 1 — ATTACK REPORT
#  Panels kept after deduplication vs Figs 1-10:
#    ✅ Shot Comparison tiles  (Fig 4 = bars; tiles = different visual)
#    ✅ Danger Creation Home   (not in 1-10)
#    ✅ Danger Creation Away   (not in 1-10)
#    ✅ GK Saves               (not in 1-10)
#    ✅ Zone 14 Home           (not in 1-10)
#    ✅ Zone 14 Away           (not in 1-10)
#    ✅ xG / xGoT tiles        (different format from Fig 4)
#    ❌ Shot Map mini          → REMOVED  (= Figs 2 & 3)
#    ❌ Goals Table            → REMOVED  (= Fig 4)
# ═══════════════════════════════════════════════════════════════════


def draw_match_report_p1(fig, events, info, xg_data, status):
    fig.clear()
    fig.patch.set_facecolor(BG_DARK)
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]
    hg = xg_data.get(hn, {}).get("goals", 0)
    ag = xg_data.get(an, {}).get("goals", 0)
    hxg = xg_data.get(hn, {}).get("xG", 0.0)
    axg = xg_data.get(an, {}).get("xG", 0.0)

    _page_header(
        fig,
        hn,
        an,
        hg,
        ag,
        hxg,
        axg,
        info["home_form"],
        info["away_form"],
        info["venue"],
        status,
        1,
        "⚽ ATTACK REPORT",
    )

    gs = GridSpec(
        4,
        3,
        figure=fig,
        left=0.04,
        right=0.97,
        top=0.945,
        bottom=0.030,
        hspace=0.65,
        wspace=0.13,
        height_ratios=[0.70, 1.15, 1.00, 0.55],
    )

    # ── Row 0: Shot Comparison tiles (full width) ──────────────────
    # ✅ UNIQUE — Fig 4 uses bar chart; these are per-metric tiles
    _panel_shot_comparison(fig.add_subplot(gs[0, :]), events, info, xg_data)

    # ── Row 1: Danger Home | GK Saves | Danger Away ────────────────
    # ✅ UNIQUE — none of these exist in Figs 1-10
    _panel_danger(fig.add_subplot(gs[1, 0]), events, hid, C_RED, hn)
    _panel_gk_saves(fig.add_subplot(gs[1, 1]), events, info)
    _panel_danger(fig.add_subplot(gs[1, 2]), events, aid, C_BLUE, an)

    # ── Row 2: Zone 14 Home | Avg Position Home+Away | Zone 14 Away ─
    # ✅ UNIQUE — not in 1-10
    _panel_zone14(fig.add_subplot(gs[2, 0]), events, hid, C_RED, hn)
    _panel_avg_pos_dual(fig.add_subplot(gs[2, 1]), events, info)  # both teams
    _panel_zone14(fig.add_subplot(gs[2, 2]), events, aid, C_BLUE, an)

    # ── Row 3: xG / xGoT / OnTarget tiles (full width) ─────────────
    # ✅ UNIQUE FORMAT — Fig 4 has bar chart; tiles show absolute values per metric
    _panel_xg_tiles(fig.add_subplot(gs[3, :]), events, info, xg_data)

    _watermark(fig)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 2 — POSSESSION & DEFENSE REPORT
#  Panels kept after deduplication vs Figs 1-10:
#    ✅ Match Statistics        (not in 1-10)
#    ✅ Pass Map/Thirds Home    (standalone with thirds breakdown)
#    ✅ Pass Map/Thirds Away    (standalone with thirds breakdown)
#    ✅ xT per Minute           (Fig 1 = cumulative; this = per-minute bar)
#    ✅ Progressive Home        (not in 1-10)
#    ✅ Progressive Away        (not in 1-10)
#    ✅ Crosses                 (not in 1-10)
#    ✅ Defensive HM Home       (not in 1-10)
#    ✅ Defensive HM Away       (not in 1-10)
#    ✅ Territorial Control     (not in 1-10)
#    ✅ Avg Position Home       (not in 1-10)
#    ✅ Avg Position Away       (not in 1-10)
#    ✅ Possession Donuts       (not in 1-10)
#    ❌ Pass Network mini Home  → REMOVED (= Fig 7)
#    ❌ Pass Network mini Away  → REMOVED (= Fig 8)
# ═══════════════════════════════════════════════════════════════════


def draw_match_report_p2(fig, events, info, xg_data, status):
    fig.clear()
    fig.patch.set_facecolor(BG_DARK)
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]
    hg = xg_data.get(hn, {}).get("goals", 0)
    ag = xg_data.get(an, {}).get("goals", 0)
    hxg = xg_data.get(hn, {}).get("xG", 0.0)
    axg = xg_data.get(an, {}).get("xG", 0.0)

    _page_header(
        fig,
        hn,
        an,
        hg,
        ag,
        hxg,
        axg,
        info["home_form"],
        info["away_form"],
        info["venue"],
        status,
        2,
        "🔵 POSSESSION & DEFENSE REPORT",
    )

    gs = GridSpec(
        5,
        3,
        figure=fig,
        left=0.04,
        right=0.97,
        top=0.945,
        bottom=0.030,
        hspace=0.60,
        wspace=0.13,
        height_ratios=[0.80, 1.0, 1.0, 1.0, 1.0],
    )

    # ── Row 0: Match Stats | Territorial Control | Possession Donuts ──
    # ✅ All UNIQUE — not in 1-10
    _panel_match_stats(fig.add_subplot(gs[0, 0]), events, info, xg_data)
    _panel_territorial(fig.add_subplot(gs[0, 1]), events, info)
    _panel_donut_dual(fig.add_subplot(gs[0, 2]), events, info)

    # ── Row 1: Pass Map/Thirds Home | Pass Map/Thirds Away ────────────
    # ✅ Pass thirds = NEW (Figs 5-6 have no third breakdown)
    # NOTE: xT/min panel removed by request — pass thirds now span the row equally.

    from matplotlib.gridspec import GridSpecFromSubplotSpec

    row1_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.18)
    _panel_pass_thirds(fig.add_subplot(row1_gs[0, 0]), events, hid, C_RED, hn)
    _panel_pass_thirds(fig.add_subplot(row1_gs[0, 1]), events, aid, C_BLUE, an)

    # ── Row 2: Progressive Home | Crosses (split) | Progressive Away ──
    # ✅ All UNIQUE — crosses now shown per team in side panels
    _panel_crosses_team(fig.add_subplot(gs[2, 0]), events, hid, C_RED, hn)
    _panel_progressive(fig.add_subplot(gs[2, 1]), events, hid, C_RED, hn)
    _panel_crosses_team(fig.add_subplot(gs[2, 2]), events, aid, C_BLUE, an)

    # ── Row 3: Defensive HM Home | (legend) | Defensive HM Away ──────
    # ✅ UNIQUE
    _panel_defensive_heatmap(fig.add_subplot(gs[3, 0]), events, hid, C_RED, hn, info)
    _panel_def_legend(fig.add_subplot(gs[3, 1]))
    _panel_defensive_heatmap(fig.add_subplot(gs[3, 2]), events, aid, C_BLUE, an, info)

    # ── Row 4: Avg Position Home | (separator) | Avg Position Away ────
    # ✅ UNIQUE
    _panel_avg_position(fig.add_subplot(gs[4, 0]), events, hid, C_RED, hn)
    _panel_avg_position(fig.add_subplot(gs[4, 2]), events, aid, C_BLUE, an)
    # centre: defensive action counts table
    _panel_def_counts(fig.add_subplot(gs[4, 1]), events, info)

    _watermark(fig)


# ═══════════════════════════════════════════════════════════════════
#  ADDITIONAL PANEL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _panel_zone14(ax, events, tid, tc, name):
    """Zone 14 + Half-Spaces — single team."""
    from matplotlib.patches import Rectangle as Rect

    _mini_pitch(ax)
    _lbl(ax, f"Zone 14 & Half-Spaces — {name}", tc)
    ax.add_patch(
        Rect((66, 33), 17, 34, facecolor=tc, alpha=0.22, edgecolor=tc, lw=1.5, zorder=2)
    )
    ax.add_patch(
        Rect(
            (66, 67),
            17,
            13,
            facecolor="#a855f7",
            alpha=0.22,
            edgecolor="#a855f7",
            lw=1.2,
            zorder=2,
        )
    )
    ax.add_patch(
        Rect(
            (66, 20),
            17,
            13,
            facecolor="#a855f7",
            alpha=0.22,
            edgecolor="#a855f7",
            lw=1.2,
            zorder=2,
        )
    )
    ev = events[
        (events["team_id"] == tid) & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if ev.empty:
        return
    z14 = ev[ev["x"].between(66, 83) & ev["y"].between(33, 67)]
    lhs = ev[ev["x"].between(66, 83) & ev["y"].between(67, 80)]
    rhs = ev[ev["x"].between(66, 83) & ev["y"].between(20, 33)]
    if not z14.empty:
        ax.scatter(
            z14["x"], z14["y"], c=tc, s=10, alpha=0.55, zorder=4, edgecolors="none"
        )
    for val, yx, yy, col in [
        (len(z14), 74.5, 50, tc),
        (len(lhs), 74.5, 73.5, "#a855f7"),
        (len(rhs), 74.5, 26.5, "#a855f7"),
    ]:
        ax.text(
            yx,
            yy,
            str(val),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            zorder=6,
            bbox=dict(
                boxstyle="circle,pad=0.35",
                facecolor=col,
                alpha=0.88,
                edgecolor="white",
                lw=0.9,
            ),
        )
    ax.text(
        50,
        -5.5,
        f"Zone14:{len(z14)}  L.HalfSpace:{len(lhs)}  R.HalfSpace:{len(rhs)}",
        ha="center",
        va="top",
        color=tc,
        fontsize=6.5,
        fontweight="bold",
    )


def _panel_avg_pos_dual(ax, events, info):
    """Average positions — both teams on same mini-pitch (overview)."""
    _mini_pitch(ax)
    _lbl(ax, "Avg Positions (Both Teams)", TEXT_DIM)
    for tid, tc in [(info["home_id"], C_RED), (info["away_id"], C_BLUE)]:
        ev = events[
            (events["team_id"] == tid) & events[["x", "y"]].notna().all(axis=1)
        ].copy()
        if ev.empty:
            continue
        avg = ev.groupby("player")[["x", "y"]].mean()
        cnt = ev.groupby("player").size()
        mx = cnt.max() if not cnt.empty else 1
        for pl, row in avg.iterrows():
            n = cnt.get(pl, 1)
            ax.scatter(
                row["x"],
                row["y"],
                c=tc,
                s=20 + n / mx * 55,
                edgecolors="white",
                lw=0.5,
                alpha=0.88,
                zorder=4,
            )


def _panel_donut_dual(ax, events, info):
    """Two possession donuts side-by-side inside one axes."""
    ax.set_facecolor(BG_MID)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ev = events[events[["x", "y"]].notna().all(axis=1)].copy()
    total = len(ev)
    for i, (tid, tc, name) in enumerate(
        [
            (info["home_id"], C_RED, info["home_name"]),
            (info["away_id"], C_BLUE, info["away_name"]),
        ]
    ):
        pct = round(int((ev["team_id"] == tid).sum()) / total * 100, 1) if total else 0
        # sub-axes inside this cell
        cx = 0.25 + i * 0.50
        sub = ax.inset_axes([cx - 0.20, 0.12, 0.40, 0.72])
        sub.pie(
            [pct, 100 - pct],
            colors=[tc, "#0A0A0A"],
            startangle=90,
            wedgeprops=dict(width=0.40, edgecolor=BG_DARK, lw=1.5),
        )
        is_light_report = str(BG_DARK).upper() in {"#FFFFFF", "WHITE"}
        pct_color = "#111827" if is_light_report else "white"
        pct_stroke = "white" if is_light_report else BG_DARK
        sub.text(
            0,
            0,
            f"{pct}%",
            ha="center",
            va="center",
            color=pct_color,
            fontsize=13,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.2, foreground=pct_stroke)],
        )
        sub.set_xlim(-1.3, 1.3)
        sub.set_ylim(-1.3, 1.3)
        ax.text(
            cx,
            0.07,
            name[:12],
            ha="center",
            va="top",
            color=tc,
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            cx,
            0.94,
            "Ball Touches",
            ha="center",
            va="top",
            color=TEXT_DIM,
            fontsize=7.5,
            transform=ax.transAxes,
        )


def _panel_def_legend(ax):
    """Legend for defensive heatmap types."""
    ax.set_facecolor(BG_MID)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.50,
        0.97,
        "Defensive Actions",
        ha="center",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=9,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.88,
        "Legend",
        ha="center",
        va="top",
        color=TEXT_DIM,
        fontsize=8,
        transform=ax.transAxes,
    )
    items = list(DEFENSIVE_TYPES.items())
    y = 0.78
    for dtype, (col, short) in items:
        ax.add_patch(
            plt.Circle((0.12, y), 0.04, facecolor=col, transform=ax.transAxes, zorder=3)
        )
        ax.text(
            0.20,
            y,
            dtype,
            ha="left",
            va="center",
            color=col,
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
        )
        y -= 0.11
    ax.text(
        0.50,
        0.08,
        "Hot = High density\nCool = Low density",
        ha="center",
        va="bottom",
        color=TEXT_DIM,
        fontsize=7.5,
        transform=ax.transAxes,
        style="italic",
    )


def _panel_def_counts(ax, events, info):
    """
    Defensive action summary table — both teams.
    Each row: Home count | action bar | Away count.
    Rows are evenly spaced with clear labels and colours.
    """
    ax.set_facecolor(BG_MID)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _lbl(ax, "Defensive Summary", TEXT_BRIGHT)

    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]

    n_rows = len(DEFENSIVE_TYPES)
    # Layout: headers at 0.88, separator at 0.84, rows from 0.80 down
    ROW_START = 0.80
    ROW_STEP = 0.80 / (n_rows + 0.5)

    # ── Column headers ────────────────────────────────────────────
    ax.text(
        0.22,
        0.91,
        hn[:13],
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=C_RED,
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.78,
        0.91,
        an[:13],
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=C_BLUE,
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.91,
        "Action",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=TEXT_DIM,
        fontsize=8,
    )

    # separator
    ax.plot([0.02, 0.98], [0.87, 0.87], color=GRID_COL, lw=1.0, transform=ax.transAxes)

    y = ROW_START
    for dtype, (col, short) in DEFENSIVE_TYPES.items():
        if dtype == "BlockedShot":
            h_n = _blocked_shots_for_team(events, info, hid)
            a_n = _blocked_shots_for_team(events, info, aid)
        else:
            h_n = (
                int(
                    events[
                        (events["team_id"] == hid) & (events["type"] == dtype)
                    ].shape[0]
                )
                if "type" in events.columns
                else 0
            )
            a_n = (
                int(
                    events[
                        (events["team_id"] == aid) & (events["type"] == dtype)
                    ].shape[0]
                )
                if "type" in events.columns
                else 0
            )
        tot = (h_n + a_n) or 1
        hr = h_n / tot
        bh = ROW_STEP * 0.50  # bar height in axes fraction

        # Home bar (red, left side of centre)
        ax.add_patch(
            plt.Rectangle(
                (0.34, y - bh / 2),
                (hr * 0.32),
                bh,
                facecolor=C_RED,
                alpha=0.65,
                transform=ax.transAxes,
                zorder=2,
            )
        )
        # Away bar (blue, right side of centre)
        ax.add_patch(
            plt.Rectangle(
                (0.34 + hr * 0.32, y - bh / 2),
                ((1 - hr) * 0.32),
                bh,
                facecolor=C_BLUE,
                alpha=0.65,
                transform=ax.transAxes,
                zorder=2,
            )
        )

        # Home count (left)
        ax.text(
            0.22,
            y,
            str(h_n),
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C_RED,
            fontsize=10,
            fontweight="bold",
            zorder=4,
        )
        # Action name (centre)
        ax.text(
            0.50,
            y,
            short,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=col,
            fontsize=8,
            fontweight="bold",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=BG_DARK,
                alpha=0.85,
                edgecolor=col,
                lw=0.8,
            ),
        )
        # Away count (right)
        ax.text(
            0.78,
            y,
            str(a_n),
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=C_BLUE,
            fontsize=10,
            fontweight="bold",
            zorder=4,
        )

        y -= ROW_STEP


def draw_match_report(fig, events, info, xg_data, status):
    """Legacy stub — calls page 1."""
    draw_match_report_p1(fig, events, info, xg_data, status)


# ══════════════════════════════════════════════════════════════════════
#  VERTICAL PITCH HELPER
# ══════════════════════════════════════════════════════════════════════
def _vert_pitch(ax, bg=PITCH_COL, half=False):
    """
    Vertical pitch — attacking direction UP (y=100).
    WhoScored coords: event.x → vertical (y-axis), event.y → horizontal (x-axis).
    half=True → draw only the attacking half (y 50..100 in pitch coords).
    """
    ax.set_facecolor(bg)
    # Let the narrow figure slot stretch the 100x100 legacy pitch vertically.
    ax.set_aspect("auto")
    # subtle box fills
    ax.add_patch(
        plt.Rectangle(
            (21.1, 83.5),
            57.8,
            16.5,
            facecolor="#ffffff",
            alpha=0.025,
            edgecolor="none",
            zorder=1,
        )
    )

    def l(xs, ys, lw=1.2, a=0.75, ls="-"):
        ax.plot(
            xs,
            ys,
            color="white",
            lw=lw,
            alpha=a,
            ls=ls,
            zorder=2,
            solid_capstyle="round",
        )

    y0 = 50 if half else 0
    # Outer boundary
    l([0, 100, 100, 0, 0], [y0, y0, 100, 100, y0], lw=1.9, a=0.88)
    if not half:
        l([0, 100], [50, 50], lw=1.0, a=0.40, ls="--")
        ax.add_patch(
            plt.Circle(
                (50, 50),
                9.15 * 100 / 105,
                color="white",
                fill=False,
                lw=0.9,
                alpha=0.38,
                zorder=2,
            )
        )
        ax.plot(50, 50, "o", color="white", ms=2.0, alpha=0.50, zorder=2)
    # Penalty box top
    l([21.1, 78.9, 78.9, 21.1, 21.1], [83.5, 83.5, 100, 100, 83.5], lw=1.0, a=0.58)
    # 6-yard box top
    l([36.8, 63.2, 63.2, 36.8, 36.8], [94.5, 94.5, 100, 100, 94.5], lw=0.75, a=0.42)
    # Goal top
    l([44, 56], [100, 100], lw=4.5, a=0.96)
    if not half:
        # Penalty box bottom
        l([21.1, 78.9, 78.9, 21.1, 21.1], [0, 0, 16.5, 16.5, 0], lw=1.0, a=0.45)
        l([36.8, 63.2, 63.2, 36.8, 36.8], [0, 0, 5.5, 5.5, 0], lw=0.75, a=0.35)
        l([44, 56], [0, 0], lw=4.5, a=0.85)
    # Penalty arcs
    ax.add_patch(
        matplotlib.patches.Arc(
            (50, 83.5),
            18 * 100 / 105,
            18,
            angle=0,
            theta1=0,
            theta2=180,
            color="white",
            lw=0.9,
            alpha=0.38,
            zorder=2,
        )
    )
    if not half:
        ax.add_patch(
            matplotlib.patches.Arc(
                (50, 16.5),
                18 * 100 / 105,
                18,
                angle=0,
                theta1=180,
                theta2=360,
                color="white",
                lw=0.9,
                alpha=0.32,
                zorder=2,
            )
        )
    # Penalty spots
    ax.plot(50, 89, "o", color="white", ms=1.8, alpha=0.55, zorder=2)

    xl = -3
    xr = 103
    yb = 47 if half else -5
    yt = 107
    ax.set_xlim(xl, xr)
    ax.set_ylim(yb, yt)
    ax.axis("off")


# ══════════════════════════════════════════════════════════════════════
#  PANEL 1 — Dominating Zone  (both teams, horizontal pitch)
# ══════════════════════════════════════════════════════════════════════
def _panel_dominating_zone(ax, events, info):
    """
    Horizontal pitch divided into N_COLS × N_ROWS zones.
    Home > 55% → home colour  |  Away > 55% → away colour  |  45-55% → gray (contested)
    """
    N_COLS, N_ROWS = 6, 4
    _mini_pitch(ax, bg="#060d06")

    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]

    ev = events[events[["x", "y"]].notna().all(axis=1)].copy()
    cw = 100 / N_COLS
    rh = 100 / N_ROWS

    CONTESTED = "#5a5f6b"
    ALPHA_DOM = 0.68
    ALPHA_CONT = 0.42

    for ci in range(N_COLS):
        for ri in range(N_ROWS):
            x0, x1 = ci * cw, (ci + 1) * cw
            y0, y1 = ri * rh, (ri + 1) * rh
            zone = ev[
                (ev["x"] >= x0) & (ev["x"] < x1) & (ev["y"] >= y0) & (ev["y"] < y1)
            ]
            h_n = int((zone["team_id"] == hid).sum())
            a_n = int((zone["team_id"] == aid).sum())
            tot = (h_n + a_n) or 1
            hr = h_n / tot

            if hr > 0.55:
                col, alpha = C_RED, ALPHA_DOM
            elif hr < 0.45:
                col, alpha = C_BLUE, ALPHA_DOM
            else:
                col, alpha = CONTESTED, ALPHA_CONT

            ax.add_patch(
                plt.Rectangle(
                    (x0 + 0.4, y0 + 0.4),
                    cw - 0.8,
                    rh - 0.8,
                    facecolor=col,
                    alpha=alpha,
                    edgecolor="none",
                    zorder=2,
                    transform=ax.transData,
                )
            )

    # Legend
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=C_RED, alpha=0.78, label=f"{hn[:12]}  (>55%)"),
            mpatches.Patch(
                facecolor=CONTESTED, alpha=0.65, label="Contested  (45–55%)"
            ),
            mpatches.Patch(facecolor=C_BLUE, alpha=0.78, label=f"{an[:12]}  (>55%)"),
        ],
        fontsize=7.5,
        ncol=3,
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.07),
        framealpha=0.92,
    )

    # Attacking direction labels
    ax.text(
        1,
        -5.5,
        f"← {hn} Attacks",
        ha="left",
        va="top",
        color=C_RED,
        fontsize=7,
        fontweight="bold",
    )
    ax.text(
        99,
        -5.5,
        f"{an} Attacks →",
        ha="right",
        va="top",
        color=C_BLUE,
        fontsize=7,
        fontweight="bold",
    )


# ══════════════════════════════════════════════════════════════════════
#  PANEL 2 — Penalty Box Entries  (single team, vertical half-pitch)
# ══════════════════════════════════════════════════════════════════════
def _panel_box_entries(ax, events, tid, tc, name):
    """
    Passes & carries that START outside the penalty box and END inside it.
    Pass  = solid arrow  |  Carry = dashed arrow
    """
    _vert_pitch(ax, half=True)
    _lbl(ax, f"Box Entries — {name}", tc)

    PBX1, PBX2 = 21.1, 78.9  # y on vertical pitch (= event.y transposed)
    PBY1 = 83.5  # x on vertical pitch (= event.x)

    def in_box(ex, ey):
        return ex >= PBY1 and PBX1 <= ey <= PBX2

    def _vx(ey):
        return float(ey)  # event.y  → ax x

    def _vy(ex):
        return float(ex)  # event.x  → ax y

    # Passes
    passes = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    pass_entries = passes[
        passes.apply(
            lambda r: not in_box(r["x"], r["y"]) and in_box(r["end_x"], r["end_y"]),
            axis=1,
        )
    ]

    # Carries (type == "Carry" if present)
    carry_entries = pd.DataFrame()
    if "type" in events.columns:
        carries = events[
            (events["type"] == "Carry")
            & (events["team_id"] == tid)
            & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        ].copy()
        carry_entries = carries[
            carries.apply(
                lambda r: not in_box(r["x"], r["y"]) and in_box(r["end_x"], r["end_y"]),
                axis=1,
            )
        ]

    # Draw passes
    for _, r in pass_entries.iterrows():
        ax.annotate(
            "",
            xy=(_vx(r["end_y"]), _vy(r["end_x"])),
            xytext=(_vx(r["y"]), _vy(r["x"])),
            arrowprops=dict(
                arrowstyle="-|>", color=tc, lw=1.5, alpha=0.88, mutation_scale=8
            ),
            zorder=5,
        )
    # Draw carries
    for _, r in carry_entries.iterrows():
        ax.annotate(
            "",
            xy=(_vx(r["end_y"]), _vy(r["end_x"])),
            xytext=(_vx(r["y"]), _vy(r["x"])),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#c084fc",
                lw=1.3,
                alpha=0.82,
                linestyle="dashed",
                mutation_scale=7,
            ),
            zorder=5,
        )

    # End-point dots inside box
    for _, r in pass_entries.iterrows():
        ax.scatter(
            _vx(r["end_y"]),
            _vy(r["end_x"]),
            c=tc,
            s=22,
            edgecolors="white",
            lw=0.6,
            zorder=6,
        )
    for _, r in carry_entries.iterrows():
        ax.scatter(
            _vx(r["end_y"]),
            _vy(r["end_x"]),
            c="#c084fc",
            s=18,
            edgecolors="white",
            lw=0.5,
            zorder=6,
        )

    # Entry-side breakdown (left / mid / right based on end_y)
    n_left = int(
        ((pass_entries["end_y"] < 35)).sum()
        + (len(carry_entries) and (carry_entries["end_y"] < 35).sum() or 0)
    )
    n_right = int(
        ((pass_entries["end_y"] > 65)).sum()
        + (len(carry_entries) and (carry_entries["end_y"] > 65).sum() or 0)
    )
    n_mid = (len(pass_entries) + len(carry_entries)) - n_left - n_right

    for xp, lbl_, val in [
        (15, "Left", n_left),
        (50, "Mid", n_mid),
        (85, "Right", n_right),
    ]:
        ax.text(
            xp,
            48.5,
            f"{lbl_}\n{val}",
            ha="center",
            va="top",
            color=tc,
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=BG_DARK,
                alpha=0.80,
                edgecolor="none",
            ),
        )

    # Stats line
    n_p = len(pass_entries)
    n_c = len(carry_entries)
    ax.text(
        50,
        106,
        f"Total: {n_p + n_c}   By Pass: {n_p}   By Carry: {n_c}",
        ha="center",
        va="bottom",
        color=TEXT_DIM,
        fontsize=7,
    )

    ax.legend(
        handles=[
            Line2D([0], [0], color=tc, lw=1.5, label=f"Pass ({n_p})"),
            Line2D(
                [0],
                [0],
                color="#c084fc",
                lw=1.3,
                linestyle="dashed",
                label=f"Carry ({n_c})",
            ),
        ],
        fontsize=7,
        facecolor=BG_MID,
        edgecolor="none",
        labelcolor="white",
        loc="lower right",
        framealpha=0.85,
    )


# ══════════════════════════════════════════════════════════════════════
#  PANEL 3 — Progressive Carries  (single team, vertical pitch)
# ══════════════════════════════════════════════════════════════════════
def _panel_prog_carries(ax, events, tid, tc, name):
    """
    Carries that advance the ball ≥10 yards (~9.5 units) toward goal.
    Exclude own defensive third (event.x < 33).
    """
    _vert_pitch(ax)
    _lbl(ax, f"Progressive Carries — {name}", tc)

    carries = pd.DataFrame()
    if "type" in events.columns:
        carries = events[
            (events["type"] == "Carry")
            & (events["team_id"] == tid)
            & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        ].copy()

    if carries.empty:
        # Fallback: approximate from consecutive same-player touch sequences
        ax.text(
            50,
            50,
            "No carry data\nin this match",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return

    # Progressive: forward distance ≥ 9.5 units, not from own defensive third
    carries["fwd"] = carries["end_x"] - carries["x"]
    prog = carries[(carries["fwd"] >= 9.5) & (carries["x"] >= 33)].copy()

    if prog.empty:
        ax.text(
            50,
            50,
            "No progressive\ncarries found",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return

    def _vx(ey):
        return float(ey)

    def _vy(ex):
        return float(ex)

    # Zone colours by origin third
    def zone_col(ex):
        if ex < 50:
            return "#64748b"
        if ex < 66:
            return C_GOLD
        return C_GREEN

    for _, r in prog.iterrows():
        col_ = zone_col(r["x"])
        # Glow
        ax.annotate(
            "",
            xy=(_vx(r["end_y"]), _vy(r["end_x"])),
            xytext=(_vx(r["y"]), _vy(r["x"])),
            arrowprops=dict(
                arrowstyle="-|>", color="white", lw=3.5, alpha=0.06, mutation_scale=10
            ),
            zorder=3,
        )
        ax.annotate(
            "",
            xy=(_vx(r["end_y"]), _vy(r["end_x"])),
            xytext=(_vx(r["y"]), _vy(r["x"])),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col_,
                lw=1.4,
                alpha=0.78,
                linestyle=(0, (4, 2)),
                mutation_scale=7,
            ),
            zorder=4,
        )
        ax.scatter(
            _vx(r["y"]),
            _vy(r["x"]),
            c=col_,
            s=12,
            alpha=0.75,
            edgecolors="none",
            zorder=5,
        )

    # Top-5 carries — player label
    if "player" in prog.columns:
        top_player = prog.groupby("player").size().idxmax()
        top_n = prog.groupby("player").size().max()
        ax.text(
            50,
            106,
            f"Most by: {_short(top_player)} ({top_n})",
            ha="center",
            va="bottom",
            color="#facc15",
            fontsize=7.5,
            fontweight="bold",
        )

    # From left / mid / right (based on event.y)
    n_left = int((prog["y"] < 33).sum())
    n_right = int((prog["y"] > 67).sum())
    n_mid = len(prog) - n_left - n_right
    tot = len(prog)

    for xp, lbl_, n, col_ in [
        (10, "From Left", n_left, tc),
        (50, "From Mid", n_mid, TEXT_DIM),
        (90, "From Right", n_right, tc),
    ]:
        ax.text(
            xp,
            -3.5,
            f"{lbl_}\n{n}",
            ha="center",
            va="top",
            color=col_,
            fontsize=7,
            fontweight="bold",
        )

    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#64748b", label="Own Half"),
            mpatches.Patch(facecolor=C_GOLD, label="Mid Third"),
            mpatches.Patch(facecolor=C_GREEN, label="Final Third"),
        ],
        fontsize=6.5,
        facecolor=BG_MID,
        edgecolor="none",
        labelcolor="white",
        loc="lower right",
        ncol=1,
        framealpha=0.85,
    )


# ══════════════════════════════════════════════════════════════════════
#  PANEL 4 — High Turnovers  (single team, vertical pitch)
# ══════════════════════════════════════════════════════════════════════
def _panel_high_turnovers(ax, events, tid, tc, name):
    """
    Possession wins (Interception / BallRecovery / Tackle won) within
    the 40m-radius zone around the opponent's goal centre.
    """
    _vert_pitch(ax)
    _lbl(ax, f"High Turnovers — {name}", tc)

    R40 = 40 / 105 * 100  # 40m in WhoScored x-units ≈ 38.1
    GCX = 100  # goal centre x in WhoScored (opponent)
    GCY = 50

    # Highlight 40m circle — light fill
    circle = plt.Circle(
        (GCY, GCX),
        R40,  # (ax_x, ax_y) after transpose
        facecolor=tc,
        alpha=0.10,
        edgecolor=tc,
        lw=1.2,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(circle)

    # High-turnover event types
    HT_TYPES = {"Interception", "BallRecovery", "Tackle", "BlockedShot", "Clearance"}
    ht_events = events[
        (events["team_id"] == tid)
        & (events["type"].isin(HT_TYPES))
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()

    # Filter: inside 40m zone
    ht_events["dist_goal"] = np.sqrt(
        ((ht_events["x"] - GCX) ** 2) + ((ht_events["y"] - GCY) ** 2)
    )
    ht_high = ht_events[ht_events["dist_goal"] <= R40].copy()

    n_total = len(ht_high)

    # Scatter
    if not ht_high.empty:
        ax.scatter(
            ht_high["y"],
            ht_high["x"],
            c=tc,
            s=45,
            edgecolors="white",
            lw=0.8,
            alpha=0.92,
            zorder=6,
        )

    # Count led to shot (within next 3 minutes of play → approximate:
    # same period, minute within +3 of turnover)
    led_shot = led_goal = 0
    if not ht_high.empty and "minute" in events.columns:
        ev_sorted = events.sort_values(["minute", "second"]).reset_index(drop=True)
        for _, row in ht_high.iterrows():
            min_ = row.get("minute", 0)
            pid = row.get("period_code", "")
            window = ev_sorted[
                (ev_sorted["team_id"] == tid)
                & (ev_sorted["minute"] >= min_)
                & (ev_sorted["minute"] <= min_ + 3)
                & (ev_sorted["period_code"] == pid)
            ]
            if window["is_shot"].any():
                led_shot += 1
            if window["is_goal"].any():
                led_goal += 1

    # Hexagon stat badges
    def _hex(ax, cx, cy, txt, val, col, size=0.085):
        """Draw a hexagon badge with label."""
        import matplotlib.patches as mp

        theta = np.linspace(0, 2 * np.pi, 7)
        hx = cx + size * np.cos(theta + np.pi / 6)
        hy = cy + size * np.sin(theta + np.pi / 6)
        ax.fill(
            hx,
            hy,
            facecolor=col,
            alpha=0.92,
            edgecolor="white",
            lw=1.0,
            zorder=8,
            transform=ax.transAxes,
        )
        ax.text(
            cx,
            cy + 0.025,
            str(val),
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=9,
        )
        ax.text(
            cx,
            cy - 0.030,
            txt,
            ha="center",
            va="center",
            color="white",
            fontsize=6.5,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=9,
        )

    _hex(ax, 0.50, 0.26, "Total", n_total, tc)
    _hex(ax, 0.35, 0.14, "Led to\nShot", led_shot, tc, size=0.075)
    _hex(ax, 0.65, 0.14, "Led to\nGoal", led_goal, tc, size=0.075)

    # 40m radius annotation
    ax.text(
        50,
        61,
        "← 40m radius →",
        ha="center",
        va="center",
        color=tc,
        fontsize=6.5,
        alpha=0.65,
        fontstyle="italic",
    )


# ══════════════════════════════════════════════════════════════════════
#  PANEL 5 — Pass Target Zones  (single team, vertical pitch grid)
# ══════════════════════════════════════════════════════════════════════
def _panel_pass_target_zones(ax, events, tid, tc, name):
    """
    5×6 grid showing % of passes received (end_x/end_y) in each zone.
    Grid rows run top→bottom (attacking end at top).
    Colour intensity scales with percentage.
    """
    _vert_pitch(ax)
    _lbl(ax, f"Pass Target Zones — {name}", tc)

    passes = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events[["end_x", "end_y"]].notna().all(axis=1)
    ].copy()

    if passes.empty:
        ax.text(
            50,
            50,
            "No pass data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8,
            style="italic",
        )
        return

    N_COLS, N_ROWS = 5, 6  # width × height
    cw = 100 / N_COLS
    rh = 100 / N_ROWS
    total = len(passes)

    # Build percentage grid
    grid = np.zeros((N_ROWS, N_COLS))
    for ci in range(N_COLS):
        for ri in range(N_ROWS):
            ex0, ex1 = (N_ROWS - 1 - ri) * rh, (N_ROWS - ri) * rh  # top = high x
            ey0, ey1 = ci * cw, (ci + 1) * cw
            n = int(
                passes[
                    (passes["end_x"] >= ex0)
                    & (passes["end_x"] < ex1)
                    & (passes["end_y"] >= ey0)
                    & (passes["end_y"] < ey1)
                ].shape[0]
            )
            grid[ri, ci] = n / total * 100 if total else 0

    # Colour map: base colour with intensity
    from matplotlib.colors import to_rgba

    base_rgb = to_rgba(tc)[:3]

    max_pct = grid.max() or 1
    for ri in range(N_ROWS):
        for ci in range(N_COLS):
            pct = grid[ri, ci]
            # ax coords: ci*cw → ey (horizontal), (5-ri)*rh → ex (vertical going up)
            ax_x0 = ci * cw
            ax_y0 = (N_ROWS - 1 - ri) * rh
            alpha = 0.10 + 0.72 * (pct / max_pct)
            ax.add_patch(
                plt.Rectangle(
                    (ax_x0 + 0.5, ax_y0 + 0.5),
                    cw - 1,
                    rh - 1,
                    facecolor=tc,
                    alpha=alpha,
                    edgecolor="#2d3748",
                    lw=0.5,
                    zorder=2,
                )
            )
            if pct >= 0.5:
                txt_col = "white"
                ax.text(
                    ax_x0 + cw / 2,
                    ax_y0 + rh / 2 + rh * 0.12,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    color=txt_col,
                    fontsize=9.5,
                    fontweight="bold",
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                )

                n_passes = int(round(pct * total / 100))
                ax.text(
                    ax_x0 + cw / 2,
                    ax_y0 + rh / 2 - rh * 0.22,
                    f"({n_passes})",
                    ha="center",
                    va="center",
                    color="#cbd5e1",
                    fontsize=7.5,
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                )

    # Scatter actual pass destinations as dim dots
    ax.scatter(
        passes["end_y"],
        passes["end_x"],
        c="#94a3b8",
        s=4,
        alpha=0.25,
        edgecolors="none",
        zorder=3,
    )

    ax.text(
        50,
        106,
        f"Total Passes: {total}",
        ha="center",
        va="bottom",
        color=TEXT_DIM,
        fontsize=7,
    )


# ══════════════════════════════════════════════════════════════════════
#  TACTICAL ANALYSIS ENGINE  (fully automatic — no API needed)
# ══════════════════════════════════════════════════════════════════════
def _collect_match_stats(info, events, xg_data):
    """Extract all key stats into a flat dict."""
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]

    def _team(tid, tname):
        ev = events[events["team_id"] == tid]

        goals = int(ev["is_goal"].sum()) if "is_goal" in ev.columns else 0
        shots_ev = (
            ev[ev["is_shot"] == True] if "is_shot" in ev.columns else pd.DataFrame()
        )
        n_shots = len(shots_ev)
        on_tgt = (
            int(shots_ev[shots_ev["type"].isin(["Goal", "SavedShot"])].shape[0])
            if not shots_ev.empty and "type" in shots_ev.columns
            else 0
        )

        passes = (
            ev[ev["is_pass"] == True] if "is_pass" in ev.columns else pd.DataFrame()
        )
        p_succ = (
            int(passes[passes["outcome"] == "Successful"].shape[0])
            if not passes.empty
            else 0
        )
        p_total = len(passes)
        p_pct = round(p_succ / p_total * 100) if p_total else 0

        prog_p = (
            int(
                passes[
                    (passes["end_x"] - passes["x"] >= 9.5) & (passes["x"] >= 33)
                ].shape[0]
            )
            if not passes.empty and "end_x" in passes.columns
            else 0
        )

        fwd_p = (
            int(passes[passes["end_x"] > passes["x"]].shape[0])
            if not passes.empty and "end_x" in passes.columns
            else 0
        )
        fwd_pct = round(fwd_p / p_total * 100) if p_total else 0

        back_p = (
            int(passes[passes["end_x"] < passes["x"]].shape[0])
            if not passes.empty and "end_x" in passes.columns
            else 0
        )

        crosses_ev = (
            ev[ev["type"] == "Cross"] if "type" in ev.columns else pd.DataFrame()
        )
        cross_succ = (
            int(crosses_ev[crosses_ev["outcome"] == "Successful"].shape[0])
            if not crosses_ev.empty
            else 0
        )
        cross_pct = round(cross_succ / len(crosses_ev) * 100) if len(crosses_ev) else 0

        def_ev = _defensive_events_for_team(events, info, tid)
        tackles = (
            int(ev[ev["type"] == "Tackle"].shape[0]) if "type" in ev.columns else 0
        )
        intercept = (
            int(ev[ev["type"] == "Interception"].shape[0])
            if "type" in ev.columns
            else 0
        )
        clearance = (
            int(ev[ev["type"] == "Clearance"].shape[0]) if "type" in ev.columns else 0
        )
        blocked = _blocked_shots_for_team(events, info, tid)
        recoveries = (
            int(ev[ev["type"] == "BallRecovery"].shape[0])
            if "type" in ev.columns
            else 0
        )

        fouls = int(ev[ev["type"] == "Foul"].shape[0]) if "type" in ev.columns else 0

        ht_ev = (
            def_ev[
                np.sqrt(((def_ev["x"] - 100) ** 2) + ((def_ev["y"] - 50) ** 2)) <= 38
            ]
            if not def_ev.empty and "x" in def_ev.columns
            else pd.DataFrame()
        )

        # Zone 14 + half-spaces for PDF tactical notes.
        # Guard both x/y columns to avoid PDF crashes if provider data is partial.
        has_xy = ("x" in ev.columns) and ("y" in ev.columns)
        z14 = (
            int(ev[(ev["x"].between(66, 83)) & (ev["y"].between(33, 67))].shape[0])
            if has_xy
            else 0
        )
        left_halfspace = (
            int(ev[(ev["x"].between(66, 83)) & (ev["y"].between(17, 33))].shape[0])
            if has_xy
            else 0
        )
        right_halfspace = (
            int(ev[(ev["x"].between(66, 83)) & (ev["y"].between(67, 83))].shape[0])
            if has_xy
            else 0
        )
        halfspace_touches = left_halfspace + right_halfspace

        touches = (
            int(
                ev[
                    ev["type"].isin(
                        [
                            "Pass",
                            "TakeOn",
                            "Carry",
                            "BallRecovery",
                            "Tackle",
                            "Interception",
                            "Clearance",
                        ]
                    )
                ].shape[0]
            )
            if "type" in ev.columns
            else 0
        )

        # touches by third
        t_def = int(ev[ev["x"] < 33]["type"].count()) if "x" in ev.columns else 0
        t_mid = (
            int(ev[ev["x"].between(33, 67)]["type"].count()) if "x" in ev.columns else 0
        )
        t_att = int(ev[ev["x"] > 67]["type"].count()) if "x" in ev.columns else 0
        tot_thirds = (t_def + t_mid + t_att) or 1
        t_def_pct = round(t_def / tot_thirds * 100)
        t_mid_pct = round(t_mid / tot_thirds * 100)
        t_att_pct = round(t_att / tot_thirds * 100)

        xg_t = xg_data.get(tname, {})

        return {
            "goals": goals,
            "shots": xg_t.get("shots", n_shots),
            "on_target": xg_t.get("on_target", on_tgt),
            "xG": round(xg_t.get("xG", 0), 2),
            "xGoT": round(xg_t.get("xGoT", 0), 2),
            "passes_total": p_total,
            "pass_accuracy": p_pct,
            "prog_passes": prog_p,
            "fwd_passes": fwd_p,
            "fwd_pct": fwd_pct,
            "back_passes": back_p,
            "crosses_total": len(crosses_ev),
            "crosses_succ": cross_succ,
            "cross_pct": cross_pct,
            "touches": touches,
            "touch_def_pct": t_def_pct,
            "touch_mid_pct": t_mid_pct,
            "touch_att_pct": t_att_pct,
            "zone14_touches": z14,
            "halfspace_touches": halfspace_touches,
            "left_halfspace_touches": left_halfspace,
            "right_halfspace_touches": right_halfspace,
            "defensive_acts": len(def_ev),
            "tackles": tackles,
            "interceptions": intercept,
            "clearances": clearance,
            "blocked_shots": blocked,
            "recoveries": recoveries,
            "fouls": fouls,
            "high_turnovers": len(ht_ev),
        }

    return {
        "home": _team(hid, hn),
        "away": _team(aid, an),
        "home_name": hn,
        "away_name": an,
        "score": info.get("score", "? - ?"),
        "venue": info.get("venue", ""),
        "date": info.get("date", ""),
        "competition": info.get("competition", ""),
    }


def _ensure_match_stats_defaults(stats: dict) -> dict:
    """Make PDF/report stat access resilient to missing or partial provider data."""
    defaults = {
        "goals": 0,
        "shots": 0,
        "on_target": 0,
        "xG": 0.0,
        "xGoT": 0.0,
        "passes_total": 0,
        "pass_accuracy": 0,
        "prog_passes": 0,
        "fwd_passes": 0,
        "fwd_pct": 0,
        "back_passes": 0,
        "crosses_total": 0,
        "crosses_succ": 0,
        "cross_pct": 0,
        "touches": 0,
        "touch_def_pct": 0,
        "touch_mid_pct": 0,
        "touch_att_pct": 0,
        "zone14_touches": 0,
        "halfspace_touches": 0,
        "left_halfspace_touches": 0,
        "right_halfspace_touches": 0,
        "defensive_acts": 0,
        "tackles": 0,
        "interceptions": 0,
        "clearances": 0,
        "blocked_shots": 0,
        "recoveries": 0,
        "fouls": 0,
        "high_turnovers": 0,
    }
    out = dict(stats or {})
    for side in ("home", "away"):
        merged = defaults.copy()
        vals = out.get(side, {}) or {}
        if isinstance(vals, dict):
            merged.update(vals)
        out[side] = merged
    out.setdefault("home_name", "Home")
    out.setdefault("away_name", "Away")
    out.setdefault("score", "? - ?")
    out.setdefault("venue", "")
    out.setdefault("date", "")
    out.setdefault("competition", "")
    return out


def _w(team, other, key, higher_is_better=True):
    """Return 'dominant'/'stronger'/'weaker' label based on stat comparison."""
    tv = team.get(key, 0)
    ov = other.get(key, 0)
    if tv == ov:
        return "equal"
    if higher_is_better:
        return "superior" if tv > ov else "inferior"
    return "superior" if tv < ov else "inferior"


def generate_tactical_analysis(info, events, xg_data):
    """
    Build a full publication-ready tactical report purely from match stats.
    No external API required.
    """
    stats = _ensure_match_stats_defaults(_collect_match_stats(info, events, xg_data))
    hn, an = stats["home_name"], stats["away_name"]
    h, a = stats["home"], stats["away"]

    score_parts = stats["score"].split("-")
    hg = int(score_parts[0].strip()) if len(score_parts) >= 2 else h["goals"]
    ag = int(score_parts[1].strip()) if len(score_parts) >= 2 else a["goals"]

    winner = hn if hg > ag else (an if ag > hg else None)
    loser = an if hg > ag else (hn if ag > hg else None)
    margin = abs(hg - ag)
    draw = hg == ag

    def _dom(hv, av, unit=""):
        """Return e.g. 'Arsenal (28) vs Chelsea (15)' string."""
        return f"{hn} ({hv}{unit}) vs {an} ({av}{unit})"

    def _better(hv, av, hn_=hn, an_=an):
        return hn_ if hv > av else (an_ if av > hv else "Both sides equally")

    def _pct_diff(a_, b_):
        if b_ == 0:
            return 0
        return round((a_ - b_) / b_ * 100)

    # ── MATCH OVERVIEW ─────────────────────────────────────────────
    if draw:
        result_line = (
            f"The match between {hn} and {an} ended in a {hg}–{ag} draw, "
            "a scoreline that suggested relative parity between the two sides over 90 minutes."
        )
    else:
        result_line = (
            f"{winner} defeated {loser} {hg}–{ag}"
            f"{' in a comprehensive victory' if margin >= 3 else ' in a hard-fought contest' if margin == 1 else ''}."
        )

    xg_narrative = ""
    if h["xG"] > 0 or a["xG"] > 0:
        xg_winner = _better(h["xG"], a["xG"])
        xg_narrative = (
            f" The xG figures — {hn}: {h['xG']}, {an}: {a['xG']} — "
            f"indicate that {xg_winner} created the higher-quality opportunities."
        )

    pass_narrative = (
        f" In terms of ball circulation, {hn} recorded {h['pass_accuracy']}% pass accuracy "
        f"versus {an}'s {a['pass_accuracy']}%, "
        f"with {_better(h['prog_passes'], a['prog_passes'])} generating more forward momentum "
        f"through progressive passes ({_dom(h['prog_passes'], a['prog_passes'])})."
    )

    overview = (
        f"{result_line}{xg_narrative}{pass_narrative}\n\n"
        f"Territorially, both teams showed distinct patterns of play. "
        f"{_better(h['touch_att_pct'], a['touch_att_pct'])} operated with greater presence in the attacking "
        f"third ({_dom(h['touch_att_pct'], a['touch_att_pct'], unit='%')} of touches), "
        f"while {_better(h['defensive_acts'], a['defensive_acts'])} recorded more defensive interventions "
        f"({_dom(h['defensive_acts'], a['defensive_acts'])}).\n\n"
        f"The tactical battle was shaped by pressing intensity, positional discipline, and the ability "
        f"to exploit transitional moments. "
        f"{'High turnovers were a key feature, with ' + _better(h['high_turnovers'], a['high_turnovers']) + ' winning the ball higher up the pitch more frequently.' if max(h['high_turnovers'], a['high_turnovers']) > 2 else 'Both sides were largely organised in their defensive structure, limiting high-press opportunities.'}"
    )

    # ── xG & SHOOTING ANALYSIS ─────────────────────────────────────
    conv_h = round(h["goals"] / h["shots"] * 100) if h["shots"] else 0
    conv_a = round(a["goals"] / a["shots"] * 100) if a["shots"] else 0
    acc_h = round(h["on_target"] / h["shots"] * 100) if h["shots"] else 0
    acc_a = round(a["on_target"] / a["shots"] * 100) if a["shots"] else 0

    xg_shooting = (
        f"{hn} generated an xG of {h['xG']} from {h['shots']} shots, with {h['on_target']} on target "
        f"(accuracy: {acc_h}%) and a conversion rate of {conv_h}%. "
        f"{an} registered {a['shots']} shots producing {a['xG']} xG, with {a['on_target']} on target "
        f"(accuracy: {acc_a}%) and a conversion of {conv_a}%.\n\n"
        f"In terms of xGoT (expected goals on target), {hn} recorded {h['xGoT']} versus {an}'s {a['xGoT']}. "
        f"{'This suggests ' + hn + ' generated better-quality shots that troubled the goalkeeper more.' if h['xGoT'] > a['xGoT'] else 'This indicates ' + an + ' posed a greater threat on shots that hit the target.' if a['xGoT'] > h['xGoT'] else 'Both teams posed similar threat on their shots on target.'}\n\n"
        f"{'Finishing was the decisive factor — ' + winner + ' converted their chances efficiently, punishing ' + loser + ' for wastefulness.' if not draw and winner else 'Neither side was able to convert their xG advantage into goals consistently, contributing to the drawn outcome.' if draw else ''}"
    ).strip()

    # ── PASSING & BALL PROGRESSION ─────────────────────────────────
    passing = (
        f"{hn} completed {h['passes_total']} passes at {h['pass_accuracy']}% accuracy, with {h['fwd_pct']}% "
        f"directed forward. {an} attempted {a['passes_total']} passes at {a['pass_accuracy']}%, "
        f"with {a['fwd_pct']}% forward.\n\n"
        f"Progressive passing was a key differentiator: {_dom(h['prog_passes'], a['prog_passes'])} passes "
        f"advanced the ball at least 10 yards into the opponent's half. "
        f"{_better(h['prog_passes'], a['prog_passes'])} were more effective at pushing the ball into dangerous "
        f"areas through direct, purposeful distribution.\n\n"
        f"Back-pass volume ({_dom(h['back_passes'], a['back_passes'])}) further reveals each team's "
        f"willingness to recycle possession. "
        f"{'A high back-pass count from ' + (hn if h['back_passes'] > a['back_passes'] else an) + ' points to a more cautious, patient build-up approach.' if abs(h['back_passes'] - a['back_passes']) > 10 else 'Both teams showed broadly similar recycling tendencies.'}"
    )

    # ── PASS NETWORKS ──────────────────────────────────────────────
    pass_networks = (
        f"The pass network visualisations reveal how each team structured their ball circulation. "
        f"{_better(h['pass_accuracy'], a['pass_accuracy'])} maintained tighter positional connections, "
        f"evidenced by their superior pass completion rate ({_dom(h['pass_accuracy'], a['pass_accuracy'], unit='%')}).\n\n"
        f"Teams with compact midfield triangles and clear vertical passing lanes are typically more "
        f"effective at breaking defensive lines. "
        f"{'With ' + str(h['prog_passes']) + ' progressive passes, ' + hn + ' appeared to have more incisive midfield connectors.' if h['prog_passes'] > a['prog_passes'] else 'With ' + str(a['prog_passes']) + ' progressive passes, ' + an + ' appeared to have the more dynamic midfield engine.'}\n\n"
        f"Pass network density in the central zones highlights the key playmakers for each side. "
        f"Teams that funnelled the ball through central midfield — rather than relying on wide recycling — "
        f"tended to generate more dangerous entries into the final third."
    )

    # ── xT ANALYSIS ────────────────────────────────────────────────
    xt_analysis = (
        f"Expected Threat (xT) measures the likelihood of scoring from each ball action, rewarding passes "
        f"and carries that advance into more dangerous zones. "
        f"The xT per-minute chart captures momentum shifts throughout the match.\n\n"
        f"{_better(h['prog_passes'], a['prog_passes'])} accumulated higher xT through a greater number of "
        f"progressive ball actions ({_dom(h['prog_passes'], a['prog_passes'])} progressive passes). "
        f"This translates to sustained threatening presence in advanced areas.\n\n"
        f"Spikes in the xT-per-minute chart often correlate with periods of sustained pressure, set-piece sequences, "
        f"or transition moments. Identifying these windows is critical for understanding when each team was most "
        f"dangerous and which tactical adjustments — including substitutions — altered the game's flow."
    )

    # ── SHOT COMPARISON ────────────────────────────────────────────
    shot_cmp = (
        f"A detailed comparison of shooting metrics underlines the attacking efficiency gap between the sides. "
        f"{hn} fired {h['shots']} attempts ({h['on_target']} on target, xG {h['xG']}), while {an} managed "
        f"{a['shots']} shots ({a['on_target']} on target, xG {a['xG']}).\n\n"
        f"Shot location quality is a key component: xG values above 0.15 per shot typically indicate "
        f"attempts from high-danger zones (inside the box, central positions). "
        f"{'With an average xG of ' + str(round(h['xG']/h['shots'],2) if h['shots'] else 0) + ' per shot, ' + hn + ' generated predominantly high-quality attempts.' if h['shots'] and h['xG']/h['shots'] > 0.12 else ''}"
        f"{'With an average xG of ' + str(round(a['xG']/a['shots'],2) if a['shots'] else 0) + ' per shot, ' + an + ' generated predominantly high-quality attempts.' if a['shots'] and a['xG']/a['shots'] > 0.12 else ''}\n\n"
        f"Blocked shots ({_dom(h['blocked_shots'], a['blocked_shots'])}) reflect how well each defence "
        f"managed to intercept shooting lanes, reducing the volume of clean strikes on goal."
    ).strip()

    # ── DANGER CREATION ────────────────────────────────────────────
    danger = (
        f"Danger creation encompasses the full range of actions that generate high-threat situations — "
        f"passes into the final third, cutbacks, through balls, and set-piece delivery.\n\n"
        f"Zone 14 touches ({_dom(h['zone14_touches'], a['zone14_touches'])}) and half-space penetration "
        f"are closely correlated with goal-scoring opportunities. "
        f"{_better(h['zone14_touches'], a['zone14_touches'])} was more active in these pivotal areas.\n\n"
        f"Box entry data ({hn} vs {an}) reveals how many times each team reached the opponent's "
        f"penalty area — the most direct measure of attacking intent converting into genuine danger."
    )

    # ── ZONE 14 & HALF-SPACES ──────────────────────────────────────
    z14_analysis = (
        f"Zone 14 — the central area directly outside the penalty box — is statistically one of the most "
        f"productive zones on the pitch for shot assists and key passes.\n\n"
        f"{hn} accumulated {h['zone14_touches']} touches in Zone 14 and the half-spaces, compared to "
        f"{an}'s {a['zone14_touches']}. "
        f"{'This represents a significant advantage for ' + _better(h['zone14_touches'], a['zone14_touches']) + ', who consistently exploited the pockets between the opposition lines.' if abs(h['zone14_touches'] - a['zone14_touches']) > 5 else 'Both teams were relatively evenly matched in their occupation of these central zones.'}\n\n"
        f"Teams that dominate Zone 14 tend to create more shooting opportunities from central positions, "
        f"where xG values are highest. Half-space penetration via overlapping runs or third-man combinations "
        f"adds an additional dimension of unpredictability to attacking play."
    )

    # ── TERRITORIAL CONTROL ────────────────────────────────────────
    territorial = (
        f"Territorial domination — measured by open-play touches in each zone — reveals each team's "
        f"positional strategy over 90 minutes.\n\n"
        f"Touch distribution by third: {hn}: {h['touch_def_pct']}% defensive / {h['touch_mid_pct']}% midfield / "
        f"{h['touch_att_pct']}% attacking. {an}: {a['touch_def_pct']}% / {a['touch_mid_pct']}% / {a['touch_att_pct']}%.\n\n"
        f"{'A high defensive-third touch percentage for ' + (hn if h['touch_def_pct'] > a['touch_def_pct'] else an) + ' may indicate sustained pressure from the opposition, or a deliberate low-block defensive strategy.' if abs(h['touch_def_pct'] - a['touch_def_pct']) > 8 else 'Both teams maintained broadly similar territorial footprints, suggesting a more balanced contest.'}"
    )

    # ── POSSESSION & TOUCHES ───────────────────────────────────────
    poss_analysis = (
        f"Ball touch maps provide a spatial picture of each team's positional tendencies. "
        f"Total touch counts — {hn}: {h['touches']} vs {an}: {a['touches']} — reflect overall ball involvement.\n\n"
        f"{_better(h['touches'], a['touches'])} demonstrated greater control of possession cycles, "
        f"recycling the ball more frequently and sustaining longer periods of ball retention.\n\n"
        f"The distribution of touches across the width of the pitch indicates tactical width. "
        f"Teams that spread touches evenly across all five channels are typically more difficult to press "
        f"and harder to defend against, as they force the opposition to cover more ground."
    )

    # ── PASS MAP BY THIRD ──────────────────────────────────────────
    pass_thirds = (
        f"Passing maps divided by pitch thirds reveal each team's build-up philosophy and the depth "
        f"at which they chose to circulate the ball.\n\n"
        f"A high volume of passes in the defensive third indicates comfort playing out from the back "
        f"under pressure. {hn}'s {h['touch_def_pct']}% defensive-third touch share versus {an}'s "
        f"{a['touch_def_pct']}% suggests {'a contrast in pressing line acceptance.' if abs(h['touch_def_pct'] - a['touch_def_pct']) > 5 else 'similar depth of possession.'}\n\n"
        f"The final-third pass volume — mirroring attacking-third touch percentages ({hn}: {h['touch_att_pct']}%, "
        f"{an}: {a['touch_att_pct']}%) — shows who was more aggressive in pushing ball circulation into "
        f"dangerous areas. {_better(h['touch_att_pct'], a['touch_att_pct'])} committed to a higher, "
        f"more intensive press line and delivery into the attacking zone."
    )

    # ── CROSSES ────────────────────────────────────────────────────
    cross_analysis = (
        f"Wide delivery was {'a feature of the game' if max(h['crosses_total'], a['crosses_total']) > 8 else 'less prominent in this match'}. "
        f"{hn} attempted {h['crosses_total']} crosses (success rate: {h['cross_pct']}%), while {an} delivered "
        f"{a['crosses_total']} (success rate: {a['cross_pct']}%).\n\n"
        f"{'Cross accuracy was notably higher for ' + _better(h['cross_pct'], a['cross_pct']) + ', suggesting better delivery quality or superior movement in the box.' if abs(h['cross_pct'] - a['cross_pct']) > 10 else 'Both sides showed similar crossing accuracy, reflecting comparable wide delivery quality.'}\n\n"
        f"The origin zones of crosses — left flank, central cutback, right flank — are critical context. "
        f"Teams delivering primarily from the right flank may expose left-back vulnerabilities in "
        f"the opposition, while cutback crosses from the byline generate statistically higher xG than "
        f"swinging crosses from deep."
    )

    # ── DEFENSIVE HEATMAP ──────────────────────────────────────────
    def_heatmap = (
        f"The defensive heatmap illustrates where each team won the ball back, providing a clear "
        f"picture of pressing intensity and defensive line height.\n\n"
        f"High-intensity pressing sides tend to cluster their defensive actions in the opponent's half "
        f"or midfield, while deeper defensive blocks show concentration in their own half. "
        f"{_better(h['high_turnovers'], a['high_turnovers'])} registered more high turnovers "
        f"({_dom(h['high_turnovers'], a['high_turnovers'])}), indicating a more proactive pressing strategy.\n\n"
        f"Defensive shape can also be inferred from action type distribution. A high tackle count "
        f"({_dom(h['tackles'], a['tackles'])}) relative to interceptions ({_dom(h['interceptions'], a['interceptions'])}) "
        f"may suggest reactive defending rather than anticipatory positioning."
    )

    # ── DEFENSIVE SUMMARY ──────────────────────────────────────────
    def_summary = (
        f"Defensive performance metrics: {hn} recorded {h['tackles']} tackles, {h['interceptions']} interceptions, "
        f"{h['clearances']} clearances, {h['blocked_shots']} blocked shots, and {h['recoveries']} ball recoveries. "
        f"{an} posted {a['tackles']} / {a['interceptions']} / {a['clearances']} / {a['blocked_shots']} / {a['recoveries']} respectively.\n\n"
        f"Total defensive actions: {_dom(h['defensive_acts'], a['defensive_acts'])}. "
        f"{'A higher defensive action count for ' + _better(h['defensive_acts'], a['defensive_acts']) + ' could indicate sustained pressure absorbed, or alternatively a more aggressive pressing style.' if abs(h['defensive_acts'] - a['defensive_acts']) > 15 else 'Both teams were relatively balanced in their defensive work-rate.'}\n\n"
        f"Fouls committed ({_dom(h['fouls'], a['fouls'])}) add another dimension — "
        f"{'excessive fouls from ' + (hn if h['fouls'] > a['fouls'] else an) + ' may reflect defensive desperation or difficulty tracking runners.' if max(h['fouls'], a['fouls']) > 12 else 'foul counts were manageable for both sides, suggesting disciplined defensive approach.'}"
    )

    # ── AVERAGE POSITIONS ──────────────────────────────────────────
    avg_pos = (
        f"Average position maps provide a tactical blueprint of each team's shape and structure "
        f"across the 90 minutes. They reveal team width, defensive line height, and the compactness "
        f"of the midfield block.\n\n"
        f"Teams that maintain a high defensive line — evident from centre-backs positioned above the "
        f"halfway line in the average position map — demonstrate a desire to compress space and play "
        f"an offside trap. A low block shows as defenders clustered near their own box.\n\n"
        f"Attacking players whose average positions are deep may indicate a pressing responsibility "
        f"in the team's structure, while wide forwards positioned high and narrow suggest an inverted "
        f"winger role. The spatial gaps between units — especially between defensive and midfield lines — "
        f"often reveal where the opposition chose to attack."
    )

    # ── DOMINATING ZONE ────────────────────────────────────────────
    dom_zone = (
        f"Zone domination — where a team holds over 55% of open-play touches in a given area — "
        f"reveals which side controlled the tactical landscape of the match.\n\n"
        f"{'A higher total touch count (' + str(h['touches']) + ' vs ' + str(a['touches']) + ') gave ' + _better(h['touches'], a['touches']) + ' an advantage in zone domination across multiple areas.' if abs(h['touches'] - a['touches']) > 50 else 'The relatively even touch distribution suggests a balanced contest with neither team fully dominating large areas of the pitch.'}\n\n"
        f"Domination in wide defensive areas can indicate a team's pressing triggers, while central "
        f"midfield control reflects the ability to dictate tempo. Control of the central attacking zone "
        f"— directly in front of goal — is the clearest indicator of sustained threat generation."
    )

    # ── BOX ENTRIES ────────────────────────────────────────────────
    box_entries = (
        f"Penalty box entries — passes and carries that originate outside the box and end inside it — "
        f"are one of the most direct measures of attacking penetration.\n\n"
        f"The entry channel breakdown (left / central / right) reveals directional attacking intent. "
        f"Left-channel dominance suggests exploitation of an opponent's right-back or a left-winger "
        f"excelling in carry runs into the box. Central entries typically derive from combination play "
        f"through Zone 14 and half-space movement.\n\n"
        f"Cross-referencing box entries with xG and shot volume provides a complete picture: high box "
        f"entries with low xG may indicate a lack of clinical finishing or poor shot selection once inside; "
        f"low box entries with high xG points to efficient but infrequent penetration — counter-attack efficiency."
    )

    # ── HIGH TURNOVERS ─────────────────────────────────────────────
    high_to = (
        f"High turnovers — ball recoveries within the 40-metre radius of the opponent's goal — "
        f"are a direct measure of pressing effectiveness in dangerous areas.\n\n"
        f"{hn} registered {h['high_turnovers']} high turnovers versus {an}'s {a['high_turnovers']}. "
        f"{'This clear advantage for ' + _better(h['high_turnovers'], a['high_turnovers']) + ' indicates a more aggressive, organised counter-pressing structure that threatened to quickly transition turnovers into shots.' if abs(h['high_turnovers'] - a['high_turnovers']) > 2 else 'Neither team gained a pronounced advantage in high pressing, suggesting both sides were comfortable circulating the ball away from their own goal under moderate pressure.'}\n\n"
        f"The ability to convert high turnovers into shots on goal is the ultimate test of pressing "
        f"quality. A high turnover count that does not translate into chances may reflect poor decision-making "
        f"in the transition moment — a key area for tactical improvement."
    )

    # ── PASS TARGET ZONES ──────────────────────────────────────────
    pass_target = (
        f"Pass target zone maps display where each team directed their successful passes — "
        f"revealing attacking intent, preferred delivery channels, and how deeply they targeted "
        f"ball-receivers in dangerous areas.\n\n"
        f"High concentrations in the central attacking third indicate a direct, central-focused "
        f"attack. Heavy weighting in wide zones suggests reliance on wide combinations or "
        f"overlapping fullbacks. Deep targeting — heavy zone use near the opposition box — reflects "
        f"an aggressive, vertical passing style.\n\n"
        f"Comparing {hn}'s pass target distribution with {an}'s highlights the tactical contrast: "
        f"{'the more direct side was ' + _better(h['fwd_pct'], a['fwd_pct']) + ' (' + _dom(h['fwd_pct'], a['fwd_pct'], unit='% forward pass rate') + ').' if abs(h['fwd_pct'] - a['fwd_pct']) > 5 else 'both teams showed similar forward pass tendencies, suggesting comparable directness in their approach play.'}"
    )

    # ── TACTICAL VERDICT ───────────────────────────────────────────
    if draw:
        verdict_opener = f"This {hg}–{ag} draw was a fair reflection of a competitive, evenly-matched contest."
    else:
        verdict_opener = (
            f"{winner} were the deserved winners of this {hg}–{ag} encounter"
            f"{', controlling large phases of the game and converting their dominance into goals' if margin >= 2 else ', edging a tight contest through greater clinical efficiency'}."
        )

    ht_winner = _better(h["high_turnovers"], a["high_turnovers"])
    ht_max = max(h["high_turnovers"], a["high_turnovers"])
    ht_sentence = (
        f"The pressing game was a decisive factor — {ht_winner}'s {ht_max} high turnovers "
        f"disrupted the opponent and created additional transitional opportunities."
        if ht_max > 3
        else "High pressing was not a decisive factor, with both teams showing adequate composure under pressure."
    )
    ultimately = (
        f"the result accurately reflects the statistical superiority of {winner}"
        if not draw
        else "neither team managed to translate statistical advantages into a winning goal"
    )
    match_type = (
        "a convincing display of modern tactical football."
        if not draw and margin >= 3
        else (
            "a tight tactical contest settled by fine margins."
            if not draw
            else "a balanced tactical encounter."
        )
    )
    pp_diff = abs(h["prog_passes"] - a["prog_passes"])
    pp_note = (
        f"ball progression ({_dom(h['prog_passes'], a['prog_passes'])} progressive passes)"
        if pp_diff > 5
        else "a competitive midfield battle"
    )
    phases_intro = f"by {winner or 'the leading side'} " if not draw else ""
    phases_kind = "multiple" if not draw and margin >= 2 else "critical"
    verdict = (
        f"{verdict_opener} "
        f"The key tactical battles were won {phases_intro}in {phases_kind} phases: "
        f"{pp_note},"
        f" zone 14 occupation ({_dom(h['zone14_touches'], a['zone14_touches'])} touches),"
        f" and defensive organisation ({_dom(h['defensive_acts'], a['defensive_acts'])} defensive actions).\n\n"
        f"{ht_sentence} "
        f"Ultimately, {ultimately}, "
        f"making this {match_type}"
    )

    console.print("  [green]✅ Tactical analysis generated from match data.[/green]")

    return {
        "MATCH OVERVIEW": overview,
        "xG & SHOOTING ANALYSIS": xg_shooting,
        "PASSING & BALL PROGRESSION": passing,
        "PASS NETWORKS": pass_networks,
        "xT (EXPECTED THREAT)": xt_analysis,
        "SHOT COMPARISON": shot_cmp,
        "DANGER CREATION": danger,
        "ZONE 14 & HALF-SPACES": z14_analysis,
        "TERRITORIAL CONTROL": territorial,
        "POSSESSION & TOUCHES": poss_analysis,
        "PASS MAP BY THIRD": pass_thirds,
        "CROSSES": cross_analysis,
        "DEFENSIVE HEATMAP": def_heatmap,
        "DEFENSIVE SUMMARY": def_summary,
        "AVERAGE POSITIONS": avg_pos,
        "DOMINATING ZONE": dom_zone,
        "BOX ENTRIES": box_entries,
        "HIGH TURNOVERS": high_to,
        "PASS TARGET ZONES": pass_target,
        "TACTICAL VERDICT": verdict,
    }


# ══════════════════════════════════════════════════════════════════════
#  PDF BUILDER  (tactical report)
# ══════════════════════════════════════════════════════════════════════
def extract_score(events, home_id: int, away_id: int):
    """
    Extract the correct score from events while handling own goals.

    Accept either raw dictionaries from matchCentreData["events"] or the
    pandas DataFrame returned by parse_all(). Field goals are counted for each
    team, while an own goal is credited to the opposing team.

    Return (home_goals, away_goals).
    """
    home_goals, away_goals = 0, 0

    if hasattr(events, "iterrows"):
        if events.empty:
            return 0, 0
        for _, ev in events.iterrows():
            is_goal = bool(ev.get("is_goal", False)) or (ev.get("event_type") == "Goal")
            if not is_goal:
                continue
            tid = ev.get("team_id")
            is_og = bool(ev.get("is_own_goal", False))
            if is_og:

                if tid == home_id:
                    away_goals += 1
                else:
                    home_goals += 1
            else:
                if tid == home_id:
                    home_goals += 1
                elif tid == away_id:
                    away_goals += 1
        return home_goals, away_goals

    for ev in events or []:
        ev_type = (
            ev.get("type", {}).get("displayName", "") if isinstance(ev, dict) else ""
        )
        if ev_type != "Goal":
            continue
        qualifiers = ev.get("qualifiers", [])
        q_types = {q.get("type", {}).get("displayName", "") for q in qualifiers}
        if "OwnGoal" in q_types:
            if ev.get("teamId") == home_id:
                away_goals += 1
            else:
                home_goals += 1
        else:
            if ev.get("teamId") == home_id:
                home_goals += 1
            elif ev.get("teamId") == away_id:
                away_goals += 1
    return home_goals, away_goals


def _parse_scoreline(info, xg_data, events=None):
    """
    Return the match score as two strings: home and away.

    Prefer extract_score(events), then info["score"], and finally xg_data.
    """
    if events is not None:
        try:
            hg, ag = extract_score(events, info.get("home_id"), info.get("away_id"))
            if (hg + ag) > 0 or info.get("score"):  # accept zeros if score field exists
                return str(hg), str(ag)
        except Exception:
            pass

    score_txt = str(info.get("score", "0 - 0")).replace("–", "-").replace("—", "-")
    parts = [p.strip() for p in score_txt.split("-") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]

    hn, an = info["home_name"], info["away_name"]
    return (
        str(xg_data.get(hn, {}).get("goals", 0)),
        str(xg_data.get(an, {}).get("goals", 0)),
    )


def _fmt_num(value, digits=2):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _safe_pct(num, den):
    return round((num / den) * 100) if den else 0


def _leader_name(home_val, away_val, hn, an, tie_label="Both sides"):
    if home_val > away_val:
        return hn
    if away_val > home_val:
        return an
    return tie_label


def _figure_to_rgba(src_fig):
    """Rasterise an existing matplotlib figure into an RGBA image for PDF composition."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(src_fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)


def _wrap_panel_text(text, width=48):
    import textwrap

    parts = []
    for para in str(text).split("\n"):
        para = para.strip()
        if not para:
            parts.append("")
            continue
        parts.append(textwrap.fill(para, width=width))
    return "\n\n".join(parts)


def _xt_total(events, tid):
    if "xT" not in events.columns:
        return 0.0
    subset = events[(events["team_id"] == tid) & events["xT"].notna()]
    return round(float(subset["xT"].sum()), 2) if not subset.empty else 0.0


def _pass_third_profile(events, tid):
    passes = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & events[["x", "end_x"]].notna().all(axis=1)
    ].copy()

    profile = {
        "total": len(passes),
        "def": 0,
        "mid": 0,
        "att": 0,
        "succ_def": 0,
        "succ_mid": 0,
        "succ_att": 0,
    }
    if passes.empty:
        return profile

    zones = {
        "def": passes[passes["x"] < 33],
        "mid": passes[(passes["x"] >= 33) & (passes["x"] < 66)],
        "att": passes[passes["x"] >= 66],
    }
    for key, df_ in zones.items():
        profile[key] = len(df_)
        profile[f"succ_{key}"] = (
            int((df_["outcome"] == "Successful").sum())
            if "outcome" in df_.columns
            else 0
        )
    return profile


def _progressive_profile(events, tid):
    passes = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events[["x", "end_x"]].notna().all(axis=1)
    ].copy()
    passes = (
        passes[(passes["end_x"] - passes["x"]) >= 10].copy()
        if not passes.empty
        else passes
    )

    profile = {"total": len(passes), "def": 0, "mid": 0, "att": 0}
    if passes.empty:
        return profile

    profile["def"] = int((passes["x"] < 33).sum())
    profile["mid"] = int(((passes["x"] >= 33) & (passes["x"] < 66)).sum())
    profile["att"] = int((passes["x"] >= 66).sum())
    return profile


def _cross_profile(events, tid):
    if "is_cross" not in events.columns:
        return {"total": 0, "succ": 0, "left": 0, "middle": 0, "right": 0}

    crosses = events[
        (events["is_cross"] == True)
        & (events["team_id"] == tid)
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if crosses.empty:
        return {"total": 0, "succ": 0, "left": 0, "middle": 0, "right": 0}

    left = int((crosses["y"] < 40).sum())
    right = int((crosses["y"] > 60).sum())
    middle = len(crosses) - left - right

    return {
        "total": len(crosses),
        "succ": (
            int((crosses["outcome"] == "Successful").sum())
            if "outcome" in crosses.columns
            else 0
        ),
        "left": left,
        "middle": middle,
        "right": right,
    }


def _dominant_lane(left, middle, right):
    values = {"left": left, "middle": middle, "right": right}
    top_val = max(values.values()) if values else 0
    if top_val == 0:
        return "without a clear preferred lane"
    winners = [k for k, v in values.items() if v == top_val]
    if len(winners) > 1:
        return "without a single dominant lane"
    labels = {
        "left": "down the left",
        "middle": "through the central lane",
        "right": "down the right",
    }
    return labels[winners[0]]


def _box_entry_profile(events, tid):
    pby1 = 83.5
    pbx1, pbx2 = 21.1, 78.9

    def in_box(ex, ey):
        return ex >= pby1 and pbx1 <= ey <= pbx2

    passes = events[
        (events["is_pass"] == True)
        & (events["team_id"] == tid)
        & (events["outcome"] == "Successful")
        & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    pass_entries = (
        passes[
            passes.apply(
                lambda r: (not in_box(r["x"], r["y"]))
                and in_box(r["end_x"], r["end_y"]),
                axis=1,
            )
        ]
        if not passes.empty
        else passes
    )

    carry_entries = pd.DataFrame()
    if "type" in events.columns:
        carries = events[
            (events["type"] == "Carry")
            & (events["team_id"] == tid)
            & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        ].copy()
        carry_entries = (
            carries[
                carries.apply(
                    lambda r: (not in_box(r["x"], r["y"]))
                    and in_box(r["end_x"], r["end_y"]),
                    axis=1,
                )
            ]
            if not carries.empty
            else carries
        )

    if pass_entries.empty and carry_entries.empty:
        return {"total": 0, "pass": 0, "carry": 0, "left": 0, "middle": 0, "right": 0}

    end_y = pd.concat(
        [
            pass_entries["end_y"] if not pass_entries.empty else pd.Series(dtype=float),
            (
                carry_entries["end_y"]
                if not carry_entries.empty
                else pd.Series(dtype=float)
            ),
        ],
        ignore_index=True,
    )

    left = int((end_y < 35).sum())
    right = int((end_y > 65).sum())
    middle = len(end_y) - left - right

    return {
        "total": len(pass_entries) + len(carry_entries),
        "pass": len(pass_entries),
        "carry": len(carry_entries),
        "left": left,
        "middle": middle,
        "right": right,
    }


def _high_turnover_profile(events, tid):
    if "type" not in events.columns:
        return {"total": 0, "led_shot": 0, "led_goal": 0}

    r40 = 40 / 105 * 100
    gcx, gcy = 100, 50
    ht_types = {"Interception", "BallRecovery", "Tackle", "BlockedShot", "Clearance"}

    ht_events = events[
        (events["team_id"] == tid)
        & (events["type"].isin(ht_types))
        & events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if ht_events.empty:
        return {"total": 0, "led_shot": 0, "led_goal": 0}

    ht_events["dist_goal"] = np.sqrt(
        ((ht_events["x"] - gcx) ** 2) + ((ht_events["y"] - gcy) ** 2)
    )
    ht_high = ht_events[ht_events["dist_goal"] <= r40].copy()
    if ht_high.empty:
        return {"total": 0, "led_shot": 0, "led_goal": 0}

    led_shot = 0
    led_goal = 0
    if "minute" in events.columns:
        ev_sorted = events.sort_values(["minute", "second"]).reset_index(drop=True)
        for _, row in ht_high.iterrows():
            minute = row.get("minute", 0)
            period_code = row.get("period_code", "")
            window = ev_sorted[
                (ev_sorted["team_id"] == tid)
                & (ev_sorted["minute"] >= minute)
                & (ev_sorted["minute"] <= minute + 3)
                & (ev_sorted["period_code"] == period_code)
            ]
            if "is_shot" in window.columns and window["is_shot"].any():
                led_shot += 1
            if "is_goal" in window.columns and window["is_goal"].any():
                led_goal += 1

    return {"total": len(ht_high), "led_shot": led_shot, "led_goal": led_goal}


def _build_visual_catalog(info):
    hn, an = info["home_name"], info["away_name"]
    return [
        {
            "idx": 1,
            "section": "shared",
            "team": None,
            "kind": "shared_xg_flow",
            "title": "xG Flow",
        },
        {
            "idx": 2,
            "section": "home",
            "team": "home",
            "kind": "team_shot_map",
            "title": f"{hn} Shot Map",
        },
        {
            "idx": 3,
            "section": "away",
            "team": "away",
            "kind": "team_shot_map",
            "title": f"{an} Shot Map",
        },
        {
            "idx": 4,
            "section": "shared",
            "team": None,
            "kind": "shared_shot_breakdown",
            "title": "Shot Breakdown and Goals",
        },
        {
            "idx": 5,
            "section": "home",
            "team": "home",
            "kind": "team_pass_network",
            "title": f"{hn} Pass Network",
        },
        {
            "idx": 6,
            "section": "away",
            "team": "away",
            "kind": "team_pass_network",
            "title": f"{an} Pass Network",
        },
        {
            "idx": 7,
            "section": "home",
            "team": "home",
            "kind": "team_xt_map",
            "title": f"{hn} xT Map",
        },
        {
            "idx": 8,
            "section": "away",
            "team": "away",
            "kind": "team_xt_map",
            "title": f"{an} xT Map",
        },
        {
            "idx": 9,
            "section": "shared",
            "team": None,
            "kind": "shared_shot_comparison",
            "title": "Shot Comparison",
        },
        {
            "idx": 10,
            "section": "home",
            "team": "home",
            "kind": "team_danger_creation",
            "title": f"{hn} Danger Creation",
        },
        {
            "idx": 11,
            "section": "away",
            "team": "away",
            "kind": "team_danger_creation",
            "title": f"{an} Danger Creation",
        },
        {
            "idx": 12,
            "section": "shared",
            "team": None,
            "kind": "shared_gk_saves",
            "title": "Goalkeeper Saves",
        },
        {
            "idx": 13,
            "section": "shared",
            "team": None,
            "kind": "shared_xg_tiles",
            "title": "xG and xGoT Summary",
        },
        {
            "idx": 14,
            "section": "home",
            "team": "home",
            "kind": "team_zone14",
            "title": f"{hn} Zone 14 and Half-Spaces",
        },
        {
            "idx": 15,
            "section": "away",
            "team": "away",
            "kind": "team_zone14",
            "title": f"{an} Zone 14 and Half-Spaces",
        },
        {
            "idx": 16,
            "section": "shared",
            "team": None,
            "kind": "shared_match_stats",
            "title": "Match Statistics",
        },
        {
            "idx": 17,
            "section": "shared",
            "team": None,
            "kind": "shared_territorial",
            "title": "Territorial Control",
        },
        {
            "idx": 18,
            "section": "shared",
            "team": None,
            "kind": "shared_touches",
            "title": "Ball Touches",
        },
        {
            "idx": 19,
            "section": "home",
            "team": "home",
            "kind": "team_pass_thirds",
            "title": f"{hn} Pass Map by Third",
        },
        {
            "idx": 20,
            "section": "away",
            "team": "away",
            "kind": "team_pass_thirds",
            "title": f"{an} Pass Map by Third",
        },
        {
            "idx": 21,
            "section": "shared",
            "team": None,
            "kind": "shared_xt_per_minute",
            "title": "xT per Minute",
        },
        {
            "idx": 22,
            "section": "home",
            "team": "home",
            "kind": "team_progressive_passes",
            "title": f"{hn} Progressive Passes",
        },
        {
            "idx": 23,
            "section": "away",
            "team": "away",
            "kind": "team_progressive_passes",
            "title": f"{an} Progressive Passes",
        },
        {
            "idx": 24,
            "section": "home",
            "team": "home",
            "kind": "team_crosses",
            "title": f"{hn} Crosses",
        },
        {
            "idx": 25,
            "section": "away",
            "team": "away",
            "kind": "team_crosses",
            "title": f"{an} Crosses",
        },
        {
            "idx": 26,
            "section": "home",
            "team": "home",
            "kind": "team_def_heatmap",
            "title": f"{hn} Defensive Actions",
        },
        {
            "idx": 27,
            "section": "away",
            "team": "away",
            "kind": "team_def_heatmap",
            "title": f"{an} Defensive Actions",
        },
        {
            "idx": 28,
            "section": "shared",
            "team": None,
            "kind": "shared_def_summary",
            "title": "Defensive Summary",
        },
        {
            "idx": 29,
            "section": "home",
            "team": "home",
            "kind": "team_average_positions",
            "title": f"{hn} Average Positions",
        },
        {
            "idx": 30,
            "section": "away",
            "team": "away",
            "kind": "team_average_positions",
            "title": f"{an} Average Positions",
        },
        {
            "idx": 31,
            "section": "shared",
            "team": None,
            "kind": "shared_dominating_zone",
            "title": "Dominating Zone",
        },
        {
            "idx": 32,
            "section": "home",
            "team": "home",
            "kind": "team_box_entries",
            "title": f"{hn} Box Entries",
        },
        {
            "idx": 33,
            "section": "away",
            "team": "away",
            "kind": "team_box_entries",
            "title": f"{an} Box Entries",
        },
        {
            "idx": 34,
            "section": "home",
            "team": "home",
            "kind": "team_high_turnovers",
            "title": f"{hn} High Turnovers",
        },
        {
            "idx": 35,
            "section": "away",
            "team": "away",
            "kind": "team_high_turnovers",
            "title": f"{an} High Turnovers",
        },
        {
            "idx": 36,
            "section": "home",
            "team": "home",
            "kind": "team_pass_target_zones",
            "title": f"{hn} Pass Target Zones",
        },
        {
            "idx": 37,
            "section": "away",
            "team": "away",
            "kind": "team_pass_target_zones",
            "title": f"{an} Pass Target Zones",
        },
    ]


def _fig_to_rgb_array(fig):
    """Render a matplotlib figure to a high-resolution RGB numpy array for collage boards."""
    old_dpi = None
    try:
        old_dpi = fig.dpi
        fig.set_dpi(BOARD_RENDER_DPI)
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())
        if arr.ndim == 3 and arr.shape[-1] == 4:
            return arr[..., :3].copy()
        return arr.copy()
    except Exception:
        return np.zeros((600, 900, 3), dtype=np.uint8)
    finally:
        try:
            if old_dpi is not None:
                fig.set_dpi(old_dpi)
        except Exception:
            pass


def build_visual_category_boards(figs, info, events, xg_data, ts, figs_filenames=None):
    """Build phase-based dashboard summary boards. Visuals are matched to the
    figs list by FILENAME (not the visual catalog's idx, which can drift out of
    sync with the actual save order), so each tile always shows the right chart."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    # (kind, team) → filename keyword. Robust against catalog idx drift.
    _KW = {
        ("shared_xg_flow", None): "xg_flow",
        ("shared_shot_breakdown", None): "breakdown_goals",
        ("shared_shot_comparison", None): "shot_comparison",
        ("shared_xg_tiles", None): "xg_tiles",
        ("shared_gk_saves", None): "gk_saves",
        ("shared_match_stats", None): "match_stats",
        ("shared_xt_per_minute", None): "xt_per_minute",
        ("shared_territorial", None): "possession",
        ("shared_touches", None): "possession",
        ("shared_dominating_zone", None): "dominating",
        ("shared_def_summary", None): "defensive_summary",
        ("team_shot_map", "home"): "shot_map_home",
        ("team_shot_map", "away"): "shot_map_away",
        ("team_danger_creation", "home"): "danger_home",
        ("team_danger_creation", "away"): "danger_away",
        ("team_zone14", "home"): "zone14_home",
        ("team_zone14", "away"): "zone14_away",
        ("team_box_entries", "home"): "box_entries_home",
        ("team_box_entries", "away"): "box_entries_away",
        ("team_crosses", "home"): "crosses_home",
        ("team_crosses", "away"): "crosses_away",
        ("team_pass_network", "home"): "pass_network_home",
        ("team_pass_network", "away"): "pass_network_away",
        ("team_xt_map", "home"): "xt_map_home",
        ("team_xt_map", "away"): "xt_map_away",
        ("team_pass_thirds", "home"): "pass_thirds_home",
        ("team_pass_thirds", "away"): "pass_thirds_away",
        ("team_progressive_passes", "home"): "progressive_home",
        ("team_progressive_passes", "away"): "progressive_away",
        ("team_pass_target_zones", "home"): "pass_target_home",
        ("team_pass_target_zones", "away"): "pass_target_away",
        ("team_def_heatmap", "home"): "defensive_hm_home",
        ("team_def_heatmap", "away"): "defensive_hm_away",
        ("team_average_positions", "home"): "avg_position_home",
        ("team_average_positions", "away"): "avg_position_away",
        ("team_high_turnovers", "home"): "high_turnovers_home",
        ("team_high_turnovers", "away"): "high_turnovers_away",
    }
    _fnames = [str(x or "").lower() for x in (figs_filenames or [])]

    def _fig_index(kind, team):
        kw = _KW.get((kind, team))
        if not kw:
            return None
        for i, nm in enumerate(_fnames):
            if kw in nm:
                return i
        return None

    stats = _ensure_match_stats_defaults(_collect_match_stats(info, events, xg_data))
    catalog = [m for m in _build_visual_catalog(info) if m.get("idx", 0) <= len(figs)]

    def _find_meta(kind, team=None):
        for meta in catalog:
            if meta.get("kind") == kind and meta.get("team") == team:
                return meta
        return None

    hn, an = info.get("home_name", "Home"), info.get("away_name", "Away")
    # Use the global C_RED / C_BLUE which are already set by choose_matchup_colors()
    # in main() — guarantees the board header always matches the visualization colors.
    hc = C_RED
    ac = C_BLUE
    hg = stats["home"].get("goals", 0)
    ag = stats["away"].get("goals", 0)
    score_txt = f"{hg} : {ag}"
    comp = str(info.get("competition") or "").strip()
    date = str(info.get("date") or "").strip()
    venue = str(info.get("venue") or "").strip()
    venue_line = " | ".join([x for x in [venue, comp, date] if x])
    stat_line = (
        f"Shots {stats['home'].get('shots',0)}-{stats['away'].get('shots',0)}  |  "
        f"xG {stats['home'].get('xG',0):.2f}-{stats['away'].get('xG',0):.2f}  |  "
        f"On target {stats['home'].get('on_target',0)}-{stats['away'].get('on_target',0)}"
    )

    # ── Per-team stats used by the board stat panels ──────────────────
    hid, aid = info.get("home_id"), info.get("away_id")
    try:
        from match_report import calculate_ppda as _calc_ppda
    except Exception:
        _calc_ppda = None

    def _metrics(tid):
        o = {
            "goals": 0,
            "shots": 0,
            "xg": 0.0,
            "xt": 0.0,
            "ot": 0,
            "big": 0,
            "passes": 0,
            "pass_pct": 0,
            "key": 0,
            "box": 0,
            "tackles": 0,
            "intercept": 0,
            "recoveries": 0,
            "clearances": 0,
            "ppda": None,
        }
        if events is None or events.empty:
            return o
        d = events[events["team_id"] == tid]
        ty = d["type"].astype(str) if "type" in d.columns else pd.Series([], dtype=str)
        if "is_shot" in d.columns:
            sh = d[d["is_shot"].fillna(False) == True]  # noqa: E712
            if "is_goal" in sh.columns and "scoring_team" in sh.columns:
                sh = sh[~(sh["is_goal"].fillna(False) & (sh["scoring_team"] != tid))]
        else:
            sh = d.iloc[:0]
        o["shots"] = int(len(sh))
        o["xg"] = round(float(d["xG"].fillna(0).sum()), 2) if "xG" in d.columns else 0.0
        if "xT" in d.columns:
            _xd = d
            if "is_pass" in d.columns:
                _xd = d[d["is_pass"].fillna(False) == True]  # noqa: E712
                if "outcome" in _xd.columns:
                    _xd = _xd[_xd["outcome"] == "Successful"]
            _xv = _xd["xT"].fillna(0)
            o["xt"] = round(float(_xv[_xv > 0].sum()), 2)
        else:
            o["xt"] = 0.0
        o["ot"] = (
            int(sh["shot_whoscored_type"].isin(["Goal", "SavedShot"]).sum())
            if "shot_whoscored_type" in sh.columns
            else 0
        )
        o["big"] = (
            int(sh["big_chance"].fillna(False).astype(bool).sum())
            if "big_chance" in sh.columns
            else 0
        )
        if "is_pass" in d.columns:
            ispass = d["is_pass"].fillna(False) == True  # noqa: E712
            o["passes"] = int(ispass.sum())
            pc = int((ispass & (d.get("outcome", "") == "Successful")).sum())
            o["pass_pct"] = round(100 * pc / max(o["passes"], 1))
        if "is_key_pass" in d.columns:
            o["key"] = int((d["is_key_pass"].fillna(False) == True).sum())  # noqa: E712
        o["tackles"] = int((ty == "Tackle").sum())
        o["intercept"] = int((ty == "Interception").sum())
        o["recoveries"] = int((ty == "BallRecovery").sum())
        o["clearances"] = int((ty == "Clearance").sum())
        if {"is_pass", "end_x", "end_y", "x", "y"}.issubset(d.columns):
            pp = d[
                (d["is_pass"].fillna(False) == True) & (d["outcome"] == "Successful")
            ]  # noqa: E712
            o["box"] = int(
                (
                    (pp["end_x"] >= 83)
                    & (pp["end_y"].between(21, 79))
                    & ~((pp["x"] >= 83) & (pp["y"].between(21, 79)))
                ).sum()
            )
        return o

    TS = {"home": _metrics(hid), "away": _metrics(aid)}
    if (
        events is not None
        and "is_goal" in events.columns
        and "scoring_team" in events.columns
    ):
        g = events[events["is_goal"].fillna(False)]
        if "is_penalty_shootout" in g.columns:
            g = g[~g["is_penalty_shootout"].fillna(False)]
        TS["home"]["goals"] = int((g["scoring_team"] == hid).sum())
        TS["away"]["goals"] = int((g["scoring_team"] == aid).sum())
    else:
        TS["home"]["goals"], TS["away"]["goals"] = hg, ag
    _pt = TS["home"]["passes"] + TS["away"]["passes"] or 1
    TS["home"]["poss"] = round(100 * TS["home"]["passes"] / _pt)
    TS["away"]["poss"] = round(100 * TS["away"]["passes"] / _pt)
    # Use the own-goal-aware goal counts for the board score header.
    score_txt = f"{TS['home']['goals']} : {TS['away']['goals']}"
    if _calc_ppda is not None:
        try:
            TS["home"]["ppda"] = round(
                float(_calc_ppda(events, hid, aid).get("ppda") or 0), 2
            )
            TS["away"]["ppda"] = round(
                float(_calc_ppda(events, aid, hid).get("ppda") or 0), 2
            )
        except Exception:
            pass

    # ── Phase definitions: title · theme · visuals · stat-panel rows ──
    for _s in ("home", "away"):
        TS[_s]["otp"] = round(100 * TS[_s]["ot"] / max(TS[_s]["shots"], 1))

    # Curated tactical dashboard: 8 boards, each two big readable visuals + a
    # KPI strip. Together they tell the whole story — rhythm, chance creation,
    # danger, build-up, progression, defence, territory and pressing.
    board_defs = [
        (
            "story",
            "THE STORY",
            "Match rhythm and the head-to-head numbers.",
            "#FFC23C",
            [
                ("shared_xg_flow", None, "xG Flow"),
                ("shared_match_stats", None, "Match Statistics"),
            ],
            [
                ("Goals", "goals", 0, False),
                ("xG", "xg", 2, False),
                ("Shots", "shots", 0, False),
                ("On-target", "ot", 0, False),
                ("Possession %", "poss", 0, False),
            ],
        ),
        (
            "creation",
            "CHANCE CREATION",
            "Where each side created its shots.",
            "#34D399",
            [
                ("team_shot_map", "home", f"{hn} Shot Map"),
                ("team_shot_map", "away", f"{an} Shot Map"),
            ],
            [
                ("Shots", "shots", 0, False),
                ("xG", "xg", 2, False),
                ("On-target %", "otp", 0, False),
                ("Big chances", "big", 0, False),
            ],
        ),
        (
            "danger",
            "DANGER ZONES",
            "How each side built its dangerous chances.",
            "#34D399",
            [
                ("team_danger_creation", "home", f"{hn} Danger Creation"),
                ("team_danger_creation", "away", f"{an} Danger Creation"),
            ],
            [
                ("Shots", "shots", 0, False),
                ("Big chances", "big", 0, False),
                ("Box entries", "box", 0, False),
                ("xG", "xg", 2, False),
            ],
        ),
        (
            "buildup",
            "BUILD-UP",
            "Passing structure and circulation.",
            "#60A5FA",
            [
                ("team_pass_network", "home", f"{hn} Pass Network"),
                ("team_pass_network", "away", f"{an} Pass Network"),
            ],
            [
                ("Passes", "passes", 0, False),
                ("Pass %", "pass_pct", 0, False),
                ("Key passes", "key", 0, False),
                ("Possession %", "poss", 0, False),
            ],
        ),
        (
            "progression",
            "PROGRESSION & THREAT",
            "Who moved the ball into danger.",
            "#60A5FA",
            [
                ("team_xt_map", "home", f"{hn} xT Map"),
                ("team_xt_map", "away", f"{an} xT Map"),
            ],
            [
                ("xT", "xt", 2, False),
                ("Passes", "passes", 0, False),
                ("Key passes", "key", 0, False),
                ("Possession %", "poss", 0, False),
            ],
        ),
        (
            "defence",
            "DEFENCE & PRESSING",
            "How each side defended its goal.",
            "#F87171",
            [
                ("team_def_heatmap", "home", f"{hn} Defensive Actions"),
                ("team_def_heatmap", "away", f"{an} Defensive Actions"),
            ],
            [
                ("Tackles", "tackles", 0, False),
                ("Interceptions", "intercept", 0, False),
                ("Recoveries", "recoveries", 0, False),
                ("PPDA", "ppda", 2, True),
            ],
        ),
        (
            "territory",
            "TERRITORY & CONTROL",
            "Who owned the ball and the space.",
            "#A78BFA",
            [
                ("shared_dominating_zone", None, "Dominating Zone"),
                ("shared_territorial", None, "Territorial Control"),
            ],
            [
                ("Possession %", "poss", 0, False),
                ("Passes", "passes", 0, False),
                ("Pass %", "pass_pct", 0, False),
                ("xT", "xt", 2, False),
            ],
        ),
        (
            "pressing",
            "PRESSING & REGAINS",
            "Where each side won the ball high.",
            "#F87171",
            [
                ("team_high_turnovers", "home", f"{hn} High Turnovers"),
                ("team_high_turnovers", "away", f"{an} High Turnovers"),
            ],
            [
                ("Recoveries", "recoveries", 0, False),
                ("Tackles", "tackles", 0, False),
                ("Interceptions", "intercept", 0, False),
                ("PPDA", "ppda", 2, True),
            ],
        ),
    ]

    boards = []
    for pkey, ptitle, psub, ptheme, specs, prows in board_defs:
        tiles = []
        _seen = set()
        for kind, team, label in specs:
            fi = _fig_index(kind, team)
            if fi is not None and fi < len(figs) and fi not in _seen:
                _seen.add(fi)
                tiles.append((fi, label))
        if not tiles:
            continue
        boards.append(
            {
                "slug": f"board_{len(boards)+1:02d}_{pkey}",
                "title": ptitle,
                "subtitle": psub,
                "theme": ptheme,
                "rows": prows,
                "tiles": tiles,
            }
        )
    n_boards_total = len(boards)

    # ── Board renderer: header + horizontal KPI strip + two BIG tiles ──
    def _draw_board(board):
        tiles = board["tiles"]
        if not tiles:
            return None
        theme = board["theme"]
        W, H = 18.0, 8.6
        fig = plt.figure(figsize=(W, H), facecolor="#000000")

        # header band
        fig.add_artist(
            mpatches.Rectangle(
                (0, 0.90),
                1,
                0.10,
                facecolor="#0d1117",
                edgecolor="none",
                transform=fig.transFigure,
                zorder=1,
            )
        )
        fig.add_artist(
            mpatches.Rectangle(
                (0, 0.90),
                0.006,
                0.10,
                facecolor=theme,
                transform=fig.transFigure,
                zorder=2,
            )
        )
        fig.text(
            0.02,
            0.955,
            board["title"],
            color=theme,
            fontsize=16,
            fontweight="bold",
            family="monospace",
            zorder=3,
        )
        fig.text(
            0.02,
            0.918,
            f"{hn}  {score_txt}  {an}   ·   {board['subtitle']}",
            color="#9fb3c8",
            fontsize=12.5,
            style="italic",
            zorder=3,
        )
        fig.text(
            0.985,
            0.94,
            f"BOARD {boards.index(board)+1} / {n_boards_total}",
            ha="right",
            color="#708090",
            fontsize=10.5,
            fontweight="bold",
            family="monospace",
            zorder=3,
        )

        # ── horizontal KPI strip ───────────────────────────────────────
        prows = board["rows"]
        nkp = len(prows)
        m = 0.02
        gap = 0.014
        cw = (1 - 2 * m - (nkp - 1) * gap) / nkp
        ky, kh = 0.795, 0.075
        for i, (lbl, key, dec, low) in enumerate(prows):
            cx = m + i * (cw + gap)
            ax = fig.add_axes([cx, ky, cw, kh])
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (0.01, 0.05),
                    0.98,
                    0.9,
                    boxstyle="round,pad=0.0,rounding_size=0.06",
                    facecolor="#0d1117",
                    edgecolor="#2A2A2A",
                    lw=1.0,
                )
            )
            ax.text(
                0.5,
                0.80,
                lbl.upper(),
                ha="center",
                color="#9fb3c8",
                fontsize=9,
                fontweight="bold",
                family="monospace",
            )
            hv = TS["home"].get(key)
            av = TS["away"].get(key)
            hv = 0 if hv is None else hv
            av = 0 if av is None else av
            _sfx = "%" if "%" in lbl else ""
            hs = (f"{hv:.2f}" if dec else str(int(hv))) + _sfx
            as_ = (f"{av:.2f}" if dec else str(int(av))) + _sfx
            hw = None if hv == av else ((hv < av) if low else (hv > av))
            ax.text(
                0.30,
                0.40,
                hs,
                ha="center",
                color=(C_GOLD if hw is True else "#e8edf2"),
                fontsize=19,
                fontweight="bold",
                family="monospace",
            )
            ax.text(
                0.70,
                0.40,
                as_,
                ha="center",
                color=(C_GOLD if hw is False else "#e8edf2"),
                fontsize=19,
                fontweight="bold",
                family="monospace",
            )
            tot = (hv + av) or 1
            frac = max(0.0, min(1.0, hv / tot))
            b = ax.inset_axes([0.12, 0.12, 0.76, 0.09])
            b.set_axis_off()
            b.set_xlim(0, 1)
            b.set_ylim(0, 1)
            b.barh(0.5, frac, height=1, color=hc)
            b.barh(0.5, 1 - frac, left=frac, height=1, color=ac)
        # team labels on the strip ends
        fig.text(
            m + 0.004,
            0.878,
            hn.upper()[:16],
            color=hc,
            fontsize=8.5,
            fontweight="bold",
            family="monospace",
        )
        fig.text(
            1 - m - 0.004,
            0.878,
            an.upper()[:16],
            ha="right",
            color=ac,
            fontsize=8.5,
            fontweight="bold",
            family="monospace",
        )

        # ── two BIG tiles side by side, aspect PRESERVED (no stretch) ──
        GX, GY, GW, GH = 0.02, 0.03, 0.96, 0.72
        cols = min(2, len(tiles))
        sw = (GW - (cols - 1) * 0.02) / cols
        for idx, (fi, label) in enumerate(tiles[:2]):
            slot_x = GX + idx * (sw + 0.02)
            slot_y = GY
            img = _fig_to_rgb_array(figs[fi])
            ih, iw = img.shape[0], img.shape[1]
            ar = (iw / ih) if ih else 1.6
            w = sw
            h = (w * W) / (ar * H)
            if h > GH:
                h = GH
                w = (h * H * ar) / W
            ax_x = slot_x + (sw - w) / 2
            ax_y = slot_y + (GH - h) / 2
            ax = fig.add_axes([ax_x, ax_y, w, h], zorder=2)
            ax.imshow(img, aspect="auto")
            del img
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#2A2A2A")
                s.set_linewidth(1.0)
            fig.add_artist(
                mpatches.FancyBboxPatch(
                    (ax_x, ax_y + h - 0.001),
                    min(0.24, w * 0.5),
                    0.032,
                    boxstyle="round,pad=0.0,rounding_size=0.004",
                    transform=fig.transFigure,
                    facecolor="#0d1117",
                    edgecolor="#2A2A2A",
                    lw=0.8,
                    zorder=4,
                )
            )
            fig.add_artist(
                mpatches.Rectangle(
                    (ax_x, ax_y + h - 0.001),
                    0.004,
                    0.032,
                    facecolor=theme,
                    transform=fig.transFigure,
                    zorder=5,
                )
            )
            fig.text(
                ax_x + 0.013,
                ax_y + h + 0.015,
                label.upper(),
                color=theme,
                fontsize=10,
                fontweight="bold",
                family="monospace",
                va="center",
                zorder=6,
            )

        if venue_line:
            fig.text(
                0.5,
                0.012,
                venue_line,
                ha="center",
                color="#708090",
                fontsize=8.5,
                family="monospace",
                zorder=3,
            )
        return fig

    saved_paths = []
    board_names = []

    # Each board is built + saved under its own guard so a single board's
    # failure (e.g. an oversized canvas or a missing source fig) can't abort
    # the whole loop and silently drop every later board — the bug that left
    # only board_01 on disk. On failure we print the exact slug + error and
    # keep going.
    for group in boards:
        try:
            fig = _draw_board(group)
            if fig is None:
                console.print(
                    f"[yellow]  ⚠ Board '{group['slug']}' skipped — no matching "
                    f"source visuals.[/yellow]"
                )
                continue
            out_path = os.path.join(SAVE_DIR, f"{group['slug']}_{ts}.png")
            # Retry the save at progressively lower DPI so a single board never
            # drops out under memory pressure (the cause of "only N of 8 boards").
            _saved = False
            for _bdpi in (180, 150, 120, 100):
                try:
                    fig.savefig(
                        out_path,
                        dpi=_bdpi,
                        bbox_inches="tight",
                        facecolor="#000000",
                        edgecolor="none",
                    )
                    _saved = True
                    break
                except Exception as _sv_err:
                    import gc as _g

                    _g.collect()
                    if _bdpi == 100:
                        console.print(
                            f"[yellow]  ⚠ Board '{group['slug']}' save failed "
                            f"even at low DPI: {_sv_err}[/yellow]"
                        )
            if _saved:
                saved_paths.append(out_path)
                board_names.append(group["title"])
        except Exception as _bd_err:
            import traceback as _bd_tb

            console.print(f"[red]  ✗ Board '{group['slug']}' failed: {_bd_err}[/red]")
            console.print(f"[dim]{_bd_tb.format_exc()}[/dim]")
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass
            # Reclaim the board canvas + tiled buffers before building the next
            # (memory-heavy) board, so peak RAM stays bounded across the loop.
            import gc

            gc.collect()

    console.print(
        f"[bold green]✅ {len(saved_paths)} Summary Visuals saved:[/bold green]"
    )
    for n, name in enumerate(board_names, 1):
        console.print(f"  {n:02d} — {name}")

    return saved_paths


def _shared_section_summary(info, stats, events):
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    hg, ag = _parse_scoreline(
        info,
        {hn: {"goals": h["goals"]}, an: {"goals": a["goals"]}},
        events=events,
    )
    hxt = _fmt_num(_xt_total(events, info["home_id"]), 2)
    axt = _fmt_num(_xt_total(events, info["away_id"]), 2)

    if str(hg) == str(ag):
        opening = (
            f"{hn} and {an} drew {hg}-{ag}, but the metrics reveal a clear "
            "difference in performance."
        )
    else:
        try:
            winner = hn if int(float(hg or 0)) > int(float(ag or 0)) else an
        except Exception:
            winner = hn
        opening = f"{winner} won {hg}-{ag}, and the metrics explain the result."

    leader_xg = _leader_name(h["xG"], a["xG"], hn, an)
    leader_prog = _leader_name(h["prog_passes"], a["prog_passes"], hn, an)
    try:
        leader_xt = _leader_name(float(hxt), float(axt), hn, an)
    except Exception:
        leader_xt = hn

    line1 = (
        f"Shots {h['shots']}-{a['shots']} | xG {h['xG']}-{a['xG']} | "
        f"On target {h['on_target']}-{a['on_target']} | "
        f"Progressive passes {h['prog_passes']}-{a['prog_passes']} | "
        f"xT {hxt}-{axt}"
    )

    return (
        f"{opening}\n\n"
        f"{line1}\n\n"
        f"{leader_xg}'s xG advantage reflects better chances, not merely more attempts.\n"
        f"{leader_prog}'s progressive-pass lead shows who controlled the buildup.\n"
        f"{leader_xt}'s total xT lead indicates sustained access to dangerous areas.\n"
        "The following pages show where the threat originated and how the opponent responded."
    )


def _team_section_summary(side, info, stats, events):
    team = stats[side]
    other_side = "away" if side == "home" else "home"
    opp = stats[other_side]
    team_name = info[f"{side}_name"]
    opp_name = info[f"{other_side}_name"]
    xt_total = _xt_total(events, info[f"{side}_id"])
    opp_xt = _xt_total(events, info[f"{other_side}_id"])

    pass_acc = team.get("pass_pct", 0)
    shots_n = team.get("shots", 0)
    xg_v = team.get("xG", 0)
    avg_xg = round(xg_v / max(shots_n, 1), 2) if shots_n else 0.0
    box_e = team.get("box_entries", 0)
    prog = team.get("prog_passes", 0)
    kp = team.get("key_passes", 0)
    def_acts = team.get("defensive_acts", 0)

    line1 = (
        f"Shots {shots_n} | On target {team['on_target']} | Goals {team['goals']} | "
        f"xG {xg_v} ({avg_xg:.2f}/shot) | xT {_fmt_num(xt_total, 2)} | "
        f"Progressive passes {prog} | Box entries {box_e}"
    )

    xg_compare = "led" if xg_v > opp.get("xG", 0) else "trailed"
    xt_compare = "led" if xt_total > opp_xt else "trailed"

    return (
        f"Analysis of {team_name} against {opp_name}.\n\n"
        f"{line1}\n\n"
        f"{team_name} {xg_compare} in xG ({xg_v} versus {opp.get('xG', 0)}), "
        "an indicator of chance quality.\n"
        f"The team {xt_compare} in total xT ({_fmt_num(xt_total, 2)} versus "
        f"{_fmt_num(opp_xt, 2)}), showing how effectively it advanced into danger.\n"
        f"Key passes ({kp}) and box entries ({box_e}) measure how efficiently "
        "buildup became chances.\n"
        f"Defensive actions ({def_acts}) show whether recoveries were proactive "
        "and organized or reactive."
    )


def _visual_tactical_note(meta, info, events, xg_data, stats, next_meta=None):
    """Return concise English tactical commentary for each PDF visual."""
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    kind = meta.get("kind", "")

    shared_map = {
        "shared_match_stats": (
            f"Score {h['goals']}-{a['goals']} | xG {h['xG']}-{a['xG']} | Shots {h['shots']}-{a['shots']}",
            "This overview compares scoreline, shot volume, and chance quality to distinguish territorial control from true attacking efficiency.",
        ),
        "shared_xg_flow": (
            f"xG {h['xG']}-{a['xG']} | On Target {h['on_target']}-{a['on_target']}",
            "Sharp xG spikes indicate high-value chances, while flatter periods reflect sterile possession or low-quality shooting.",
        ),
        "shared_shot_breakdown": (
            f"Shots {h['shots']}-{a['shots']} | OT {h['on_target']}-{a['on_target']} | Blocked {h.get('blocked',0)}-{a.get('blocked',0)}",
            "Shot volume should be interpreted alongside accuracy and blocking rate to assess whether attacks reached dangerous zones.",
        ),
        "shared_territorial": (
            f"Final Third Touches {h['touch_att_pct']}%-{a['touch_att_pct']}% | Passes {h['passes']}-{a['passes']}",
            "Final-third presence highlights which side imposed field tilt and sustained territorial pressure.",
        ),
        "shared_xt_per_minute": (
            f"xT {h.get('xT',0):.2f}-{a.get('xT',0):.2f} | Progressive Passes {h['prog_passes']}-{a['prog_passes']}",
            "Expected Threat captures ball progression before the shot and helps identify the side generating more dangerous possession.",
        ),
    }

    if kind in shared_map:
        return shared_map[kind]

    team_name = (
        hn if meta.get("team") == "home" else an if meta.get("team") == "away" else hn
    )
    opp_name = an if team_name == hn else hn
    team = h if team_name == hn else a
    opp = a if team_name == hn else h

    statline = (
        f"{team_name}: xG {team['xG']} | Shots {team['shots']} | "
        f"On Target {team['on_target']} | Box Entries {team.get('box_entries',0)}"
    )

    note = (
        f"{team_name}'s attacking profile in this visual should be read relative to {opp_name}: "
        f"use spacing, density, and location of actions to identify creation zones, progression routes, "
        f"and whether possession translated into efficient final-third penetration."
    )

    return statline, note


def _safe_stat(d, key, default=0):
    try:
        return (d or {}).get(key, default)
    except Exception:
        return default


def _expert_tactical_commentary(
    kind,
    base_note,
    meta,
    info,
    stats,
    events,
    xg_data,
    side_key=None,
    other_key=None,
    team=None,
    opp=None,
    team_xt=None,
    opp_xt=None,
):
    """Expand each PDF note into a fuller tactical-analysis explanation."""
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    tid = info.get(f"{side_key}_id") if side_key else None
    team_name = info.get(f"{side_key}_name", "") if side_key else ""
    opp_name = info.get(f"{other_key}_name", "") if other_key else ""
    paragraphs = [str(base_note).strip()]

    if kind == "shared_xg_flow":
        paragraphs.append(
            "The important tactical detail is the timing of the xG jumps rather than the final total alone. A steep rise usually means the defending side allowed a clean shot, a central reception, a cut-back or a transition chance; a flat period means possession existed without penetration. This turns the chart into a match-rhythm tool, not just a shooting graph."
        )
        paragraphs.append(
            "A coaching staff would read this beside the progression and territory pages. If xG rises after long territorial pressure, the attacking process is repeatable. If it rises from isolated spikes, the team may have relied more on transition moments or individual finishing than stable control."
        )
    elif kind in {
        "shared_shot_breakdown",
        "shared_shot_comparison",
        "shared_xg_tiles",
        "shared_gk_saves",
    }:
        paragraphs.append(
            "The key question is not simply who shot more, but who shot from cleaner lanes. On-target shots show execution, blocked shots show defensive pressure around the ball, and xG shows whether the locations were genuinely valuable. High volume with many blocks often means the opponent protected the centre well and forced delayed releases."
        )
        paragraphs.append(
            "This is why shot quality matters more than raw volume in a one-match report. If the result is supported by xG, xGoT and central shot locations, the attacking edge is repeatable. If it is built on low-probability finishing or goalkeeper variance, the performance should be judged more cautiously."
        )
    elif kind == "shared_match_stats":
        paragraphs.append(
            "The statistics panel is the starting point, not the conclusion. Possession and pass accuracy describe control, but their tactical value depends on whether they connect to xT, progressive passes, box entries and final-third receptions. That connection tells us whether possession moved the opponent or merely circulated around the block."
        )
        paragraphs.append(
            "For a coaching interpretation, volume must be linked to efficiency. A team can pass more, but if those passes do not produce central access, pressure after loss or repeat entries into the area, the control remains sterile. The most valuable teams turn circulation into territorial pressure and then into chance quality."
        )
    elif kind in {"shared_territorial", "shared_touches", "shared_dominating_zone"}:
        paragraphs.append(
            "Territorial control explains where the game lived. Dominance in the midfield and final-third zones forces the opponent to defend for longer spells, lowers their starting positions and increases the chance of second-ball recoveries after attacks break down. Pressure often accumulates here before it appears in the shot count."
        )
        paragraphs.append(
            "The most important zones are not always the largest areas of dominance. A small edge in Zone 14 or either half-space can be more valuable than a large edge in deeper build-up zones because those pockets are where final passes, cut-backs and shooting actions are prepared."
        )
    elif kind == "shared_xt_per_minute":
        paragraphs.append(
            "Minute-by-minute xT captures attacking momentum before a shot exists. A strong xT spell can show that a team is moving the ball into dangerous zones even if the final action is blocked, overhit or delayed. That makes it a useful bridge between possession and chance creation."
        )
        paragraphs.append(
            "Repeated peaks suggest a stable route into threat; one-off peaks suggest transition attacks, broken-press moments or individual actions. The report should therefore treat xT as a measure of process, not as a direct replacement for xG."
        )
    elif kind == "shared_def_summary":
        paragraphs.append(
            "Defensive actions need location context. A high total can mean front-foot pressing, but it can also mean long periods of containment. High recoveries support aggressive control; deep recoveries point to a block protecting the penalty area and clearing second balls."
        )
        paragraphs.append(
            "Tackles, interceptions and recoveries describe different behaviours. Tackles show direct duels, interceptions show anticipation and cover shadows, while recoveries show which side controlled loose-ball moments after pressure, clearances or rebounds."
        )
    elif kind == "team_shot_map":
        avg_xg = round(
            _safe_stat(team, "xG", 0) / max(_safe_stat(team, "shots", 0), 1), 2
        )
        paragraphs.append(
            f"For {team_name}, the shot map is a map of access. Central attempts inside the box usually mean the attacking structure found the weak point of {opp_name}'s defensive line; wide or deep attempts suggest the opponent guided the attack away from the most valuable zones. The average value of about {avg_xg:.2f} xG per shot helps judge that balance."
        )
        paragraphs.append(
            f"The coaching point is repeatability. If {team_name}'s best shots came from planned combinations, cut-backs, underlaps or high regains, the process is stronger. If they came from loose balls or speculative angles, the scoreline may flatter the attacking structure."
        )
    elif kind == "team_danger_creation":
        paragraphs.append(
            f"This page connects creators and finishers. The strongest attacking sides do not only produce shots; they build chains of actions that repeatedly move the ball from preparation zones into the penalty area. If {team_name}'s danger points cluster in the same corridor, it reveals both the preferred route and the area {opp_name} must close earlier."
        )
        paragraphs.append(
            "A balanced spread of danger points indicates multi-lane threat. A narrow spread can still be effective if the overload is strong, but it becomes easier to defend once the opponent identifies the trigger and shifts the block toward that side."
        )
    elif kind == "team_zone14":
        paragraphs.append(
            f"Zone 14 and the half-spaces are the key pre-assist zones in settled attacks. When {team_name} can receive there, {opp_name}'s back line must decide whether to step out, hold the line or track the runner. That hesitation creates the window for slipped passes, through-balls and cut-backs."
        )
        paragraphs.append(
            "If the distribution is heavily tilted to one half-space, the defending side has a clear pressing trigger: lock that side, block the inside pass and force play wide. If the distribution is balanced, the block has to stay honest across the full width of the pitch."
        )
    elif kind == "team_box_entries":
        prof = (
            _box_entry_profile(events, tid)
            if tid is not None
            else {"total": 0, "pass": 0, "carry": 0, "left": 0, "middle": 0, "right": 0}
        )
        paragraphs.append(
            f"Box entries are one of the clearest measures of penetration. {team_name}'s {prof['total']} entries show how often the attack actually broke the last line of pressure and entered the zone where defensive mistakes become costly. Entries by pass suggest combination play; entries by carry suggest individual superiority or space created by rotations."
        )
        paragraphs.append(
            "The lane split matters as much as the total. If entries arrive mostly through one channel, the opponent can shift early and protect the near-side centre-back. If they arrive from left, centre and right, the defensive line has to handle multiple angles and loses reference points."
        )
    elif kind == "team_crosses":
        prof = (
            _cross_profile(events, tid)
            if tid is not None
            else {"total": 0, "succ": 0, "left": 0, "middle": 0, "right": 0}
        )
        acc = _safe_pct(prof["succ"], prof["total"])
        paragraphs.append(
            f"Crossing volume only has tactical value when it matches box occupation. {team_name}'s success rate of {acc}% should be read against the number of runners attacking the six-yard line, penalty spot and far post. A low rate can mean poor delivery, but it can also mean the box was underloaded or the cross was forced too early."
        )
        paragraphs.append(
            "The delivery side reveals the attacking plan. Repeated crosses from one flank indicate a deliberate wide overload, but if the opponent wins first contact comfortably, the better solution may be cut-backs, delayed arrivals or switches before the delivery."
        )
    elif kind == "team_pass_network":
        paragraphs.append(
            f"The pass network shows the skeleton of {team_name}'s possession. The most connected players are usually the structural references: centre-backs starting build-up, pivots offering the reset, or attacking midfielders linking midfield to the front line. If the network depends too heavily on one hub, {opp_name} can press that player and disturb the rhythm."
        )
        paragraphs.append(
            "Distances between nodes matter. A compact network helps short combinations and counter-pressing after loss; a stretched network can create progression lanes, but it also increases turnover risk if the first forward pass is not secure."
        )
    elif kind == "team_pass_thirds":
        paragraphs.append(
            f"The third-by-third passing split shows where {team_name}'s possession settled. A high defensive-third share means the build-up had to begin under pressure or reset often. A high midfield share points to circulation and control. A high final-third share means the team could sustain attacks after crossing halfway."
        )
        paragraphs.append(
            f"For tactical evaluation, final-third passing is usually decisive. Reaching the final third once is not enough; sustained passing there forces {opp_name} to defend multiple actions, increases the chance of a mistake and improves the odds of winning second balls around the box."
        )
    elif kind == "team_progressive_passes":
        paragraphs.append(
            f"Progressive passes describe how {team_name} advanced through the pitch. Progression from the defensive third shows bravery and line-breaking quality; progression from midfield shows control between lines; progression in the final third often leads directly into chance creation."
        )
        paragraphs.append(
            "The key coaching question is whether the receiver could face forward. A progressive pass into a trapped player can look positive in the data but may only transfer pressure. The best progressive actions break a line and give the receiver time to play the next pass."
        )
    elif kind == "team_xt_map":
        paragraphs.append(
            f"Expected Threat values ball movement before the shot. {team_name}'s total xT of {_fmt_num(team_xt or 0, 2)} shows how much their passing and carrying increased scoring probability across the match. This is useful when shot volume alone does not explain who was more dangerous."
        )
        paragraphs.append(
            f"The highest-xT zones are the routes {opp_name} struggled to control. If they sit in the half-spaces or around the box edge, the threat came from structure. If they appear deeper or wider, the danger may have come from switches, counters or early deliveries into space."
        )
    elif kind == "team_pass_target_zones":
        paragraphs.append(
            f"Target zones show intention: where {team_name} wanted the next receiver to be. A concentration in advanced wide zones suggests wing isolation; a central concentration suggests attempts to connect through the No. 10 space or the striker's feet."
        )
        paragraphs.append(
            f"For {opp_name}, this grid identifies the defensive priority. The busiest reception cells are the zones that need earlier pressure, tighter cover shadows or a different pressing angle to stop {team_name} entering their preferred rhythm."
        )
    elif kind == "team_average_positions":
        paragraphs.append(
            f"Average positions show occupation over time rather than a fixed formation. For {team_name}, the spacing between the defensive line, midfield line and front line tells us whether the side stayed compact enough to combine and counter-press, or whether gaps appeared between units."
        )
        paragraphs.append(
            "This is also useful for interpreting rest defence. High full-backs or wingers may improve width, but they can leave transition space behind them. An isolated pivot may make build-up too dependent on one player escaping pressure."
        )
    elif kind == "team_def_heatmap":
        paragraphs.append(
            f"The defensive heatmap shows where {team_name} had to solve problems without the ball. Actions high up the pitch indicate pressing success or counter-pressing after loss; actions close to the box point to deeper protection and longer spells of pressure from {opp_name}."
        )
        paragraphs.append(
            "The balance between tackles, interceptions and recoveries matters. Interceptions often show compactness and good cover shadows, tackles show direct duels, and recoveries show control of loose-ball moments after pressure or clearances."
        )
    elif kind == "team_high_turnovers":
        prof = (
            _high_turnover_profile(events, tid)
            if tid is not None
            else {"total": 0, "led_shot": 0, "led_goal": 0}
        )
        paragraphs.append(
            f"High turnovers are the best measure of whether the press created attacking value. {team_name}'s {prof['total']} high regains only become decisive when followed by immediate forward passes, shots or box entries before {opp_name} can reset."
        )
        paragraphs.append(
            "The conversion of turnovers into shots is the key benchmark. A press that wins the ball then plays backwards controls territory; a press that wins the ball and attacks quickly changes the game state."
        )
    else:
        paragraphs.append(
            "The tactical value of this page comes from connecting the visual pattern to the wider match context: chance quality, field position, pressure after loss and the ability to repeat the same route without becoming predictable."
        )

    return "\n\n".join(p for p in paragraphs if p)


def _section_title(section, info):
    if section == "shared":
        return "Shared Match Analysis"
    if section == "home":
        return f"Home Team Analysis - {info['home_name']}"
    return f"Away Team Analysis - {info['away_name']}"


def _section_color(section):
    if section == "shared":
        return C_GOLD
    return C_RED if section == "home" else C_BLUE


def _draw_pdf_header(fig, info, page_title, section_title, page_num, total_pages):
    """
    Inner pages: team names only (no score) in the header strip.
    """
    from matplotlib.patches import FancyBboxPatch

    hn = info["home_name"]
    an = info["away_name"]

    ax = fig.add_axes([0.0, 0.935, 1.0, 0.065])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Coloured halves
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0), 0.50, 1.0, boxstyle="square,pad=0", facecolor=C_RED, alpha=0.92
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.50, 0.0),
            0.50,
            1.0,
            boxstyle="square,pad=0",
            facecolor=C_BLUE,
            alpha=0.92,
        )
    )
    # Dark centre block
    ax.add_patch(
        FancyBboxPatch(
            (0.34, 0.0),
            0.32,
            1.0,
            boxstyle="square,pad=0",
            facecolor="#111827",
            alpha=0.97,
        )
    )

    # ── Team names only ─────────────────────────────────────────────
    ax.text(
        0.17,
        0.52,
        hn,
        ha="center",
        va="center",
        color=_text_on_color(C_RED),
        fontsize=14,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground=_stroke_on_color(C_RED))],
    )
    ax.text(
        0.83,
        0.52,
        an,
        ha="center",
        va="center",
        color=_text_on_color(C_BLUE),
        fontsize=14,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground=_stroke_on_color(C_BLUE))],
    )

    # ── Centre: section title (top) + page title (bottom) ──────────
    ax.text(
        0.50,
        0.72,
        section_title,
        ha="center",
        va="center",
        color="#FFD700",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.30,
        page_title,
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )

    # ── Sub-bar: competition | venue | page ────────────────────────
    info_ax = fig.add_axes([0.0, 0.915, 1.0, 0.020])
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis("off")
    info_ax.set_facecolor("#e8ecf0")
    info_ax.text(
        0.02,
        0.5,
        info.get("competition", ""),
        ha="left",
        va="center",
        color="#374151",
        fontsize=8,
    )
    info_ax.text(
        0.50,
        0.5,
        info.get("venue", ""),
        ha="center",
        va="center",
        color="#374151",
        fontsize=8,
    )
    info_ax.text(
        0.98,
        0.5,
        f"Page {page_num} of {total_pages}",
        ha="right",
        va="center",
        color="#374151",
        fontsize=8,
    )


def _draw_pdf_footer(fig, page_num, total_pages, center_text=""):
    fig.add_artist(
        plt.Line2D(
            [0.03, 0.97],
            [0.024, 0.024],
            transform=fig.transFigure,
            color="#ffffff",
            lw=0.8,
            alpha=0.9,
        )
    )
    fig.text(
        0.03, 0.013, CREDIT_TOOLS, ha="left", va="bottom", color="#ffffff", fontsize=7.5
    )
    if center_text:
        fig.text(
            0.50,
            0.013,
            center_text,
            ha="center",
            va="bottom",
            color="#ffffff",
            fontsize=7.5,
        )
    fig.text(
        0.97,
        0.013,
        f"{page_num}/{total_pages}",
        ha="right",
        va="bottom",
        color="#ffffff",
        fontsize=9,
        fontweight="bold",
    )


def _render_cover_page(pdf, info, stats, events, total_pages):
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    h_sc, a_sc = _parse_scoreline(
        info,
        {hn: {"goals": h["goals"]}, an: {"goals": a["goals"]}},
        events=events,
    )

    PDF_BG = "#000000"
    cover = plt.figure(figsize=(16, 9), facecolor=PDF_BG)
    cover.patch.set_facecolor(PDF_BG)

    bg_ax = cover.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.axis("off")
    bg_ax.set_facecolor(PDF_BG)

    txt_ax = cover.add_axes([0, 0, 1, 1], zorder=2)
    txt_ax.set_xlim(0, 1)
    txt_ax.set_ylim(0, 1)
    txt_ax.axis("off")

    txt_ax.text(
        0.25,
        0.60,
        hn,
        ha="center",
        va="center",
        color=C_RED,
        fontsize=28,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=5, foreground="#000")],
    )

    txt_ax.text(
        0.50,
        0.60,
        f"{h_sc}  –  {a_sc}",
        ha="center",
        va="center",
        color="#FFD700",
        fontsize=52,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=6, foreground="#000")],
    )

    txt_ax.text(
        0.75,
        0.60,
        an,
        ha="center",
        va="center",
        color=C_BLUE,
        fontsize=28,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=5, foreground="#000")],
    )

    txt_ax.text(
        0.50,
        0.44,
        "Statistical Tactical Analysis",
        ha="center",
        va="center",
        color="white",
        fontsize=18,
        fontweight="bold",
    )

    txt_ax.text(
        0.50,
        0.34,
        "By Mostafa Saad",
        ha="center",
        va="center",
        color="white",
        fontsize=20,
        fontweight="bold",
    )

    cover.text(
        0.97,
        0.018,
        f"1/{total_pages}",
        ha="right",
        va="bottom",
        color="white",
        fontsize=10,
        fontweight="bold",
        transform=cover.transFigure,
    )

    pdf.savefig(
        cover,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_BG,
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close(cover)


def _render_section_page(pdf, info, section, summary, page_num, total_pages):
    color = _section_color(section)
    title = _section_title(section, info)

    PDF_BG = "#000000"  # AMOLED black
    PDF_DIM = "#9ca3af"

    page = plt.figure(figsize=(16, 9), facecolor=PDF_BG)

    bar = page.add_axes([0.0, 0.93, 1.0, 0.07])
    bar.set_xlim(0, 1)
    bar.set_ylim(0, 1)
    bar.axis("off")
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.90))
    bar.text(
        0.50,
        0.52,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    page.text(
        0.50,
        0.72,
        title,
        ha="center",
        va="center",
        color=color,
        fontsize=28,
        fontweight="bold",
    )
    page.text(
        0.50,
        0.63,
        "The pages that follow keep the visual and the tactical explanation together on the same page.",
        ha="center",
        va="center",
        color=PDF_DIM,
        fontsize=11,
    )

    panel = page.add_axes([0.16, 0.26, 0.68, 0.28])
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.axis("off")
    panel.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#0d0d0d",
            edgecolor="#1f2937",
            lw=1.2,
            alpha=0.97,
        )
    )
    panel.text(
        0.05,
        0.86,
        "Section summary",
        ha="left",
        va="center",
        color=color,
        fontsize=11,
        fontweight="bold",
    )
    panel.text(
        0.05,
        0.74,
        _wrap_panel_text(summary, width=92),
        ha="left",
        va="top",
        color="#ffffff",
        fontsize=10.5,
    )

    _draw_pdf_footer(page, page_num, total_pages, center_text="Created by Mostafa Saad")
    pdf.savefig(
        page,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_BG,
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close(page)


def _render_visual_page(
    pdf, src_fig, info, meta, statline, commentary, page_num, total_pages, events=None
):
    PDF_BG = "#000000"  # AMOLED black

    page = plt.figure(figsize=(16, 9), facecolor=PDF_BG)
    section_title = _section_title(meta["section"], info)
    _draw_pdf_header(page, info, meta["title"], section_title, page_num, total_pages)

    img = _figure_to_rgba(src_fig)
    img_ax = page.add_axes([0.03, 0.08, 0.62, 0.82])
    img_ax.set_facecolor("#111827")  # the figure itself stays dark
    img_ax.imshow(img, interpolation="lanczos", aspect="equal")
    img_ax.axis("off")
    for spine in img_ax.spines.values():
        spine.set_visible(False)

    # ── Right panel: light card ─────────────────────────────────────
    panel_ax = page.add_axes([0.68, 0.08, 0.29, 0.82])
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, 1)
    panel_ax.axis("off")
    panel_ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#0a0a0a",
            edgecolor="#1f2937",
            lw=1.2,
            alpha=0.97,
        )
    )
    color = _section_color(meta["section"])
    panel_ax.text(
        0.06,
        0.95,
        "Tactical note",
        ha="left",
        va="top",
        color=color,
        fontsize=11,
        fontweight="bold",
    )
    panel_ax.text(
        0.06, 0.89, statline, ha="left", va="top", color="#9ca3af", fontsize=9.3
    )
    panel_ax.plot([0.06, 0.94], [0.84, 0.84], color="#1f2937", lw=1.0)
    panel_ax.text(
        0.06,
        0.81,
        _wrap_panel_text(commentary, width=44),
        ha="left",
        va="top",
        color="#ffffff",
        fontsize=10.2,
    )

    _draw_pdf_footer(page, page_num, total_pages)
    pdf.savefig(
        page,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_BG,
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close(page)


def build_tactical_pdf(figs, info, events, xg_data, ts):
    """Assemble the final tactical PDF with shared visuals first, then home, then away."""

    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = OUTPUT_IMAGE_DPI

    hn, an = info["home_name"], info["away_name"]
    stats = _ensure_match_stats_defaults(_collect_match_stats(info, events, xg_data))

    safe_hn = hn.replace(" ", "_").replace("/", "_")
    safe_an = an.replace(" ", "_").replace("/", "_")
    pdf_path = f"{SAVE_DIR}/tactical_report_{safe_hn}_vs_{safe_an}_{ts}.pdf"

    console.print("\n[bold cyan]  Writing tactical PDF report...[/bold cyan]")
    console.print(f"[bold cyan]  Building PDF: {pdf_path}[/bold cyan]")

    catalog = [m for m in _build_visual_catalog(info) if m["idx"] <= len(figs)]
    section_rank = {"shared": 0, "home": 1, "away": 2}
    ordered_catalog = sorted(
        catalog, key=lambda item: (section_rank[item["section"]], item["idx"])
    )

    section_pages = [
        ("shared", _shared_section_summary(info, stats, events)),
        ("home", _team_section_summary("home", info, stats, events)),
        ("away", _team_section_summary("away", info, stats, events)),
    ]

    total_pages = 1 + len(section_pages) + len(ordered_catalog)
    page_num = 1

    with PdfPages(pdf_path) as pdf:
        _render_cover_page(pdf, info, stats, events, total_pages)
        page_num += 1

        for section, summary in section_pages:
            _render_section_page(pdf, info, section, summary, page_num, total_pages)
            page_num += 1

            for meta in [m for m in ordered_catalog if m["section"] == section]:
                statline, commentary = _visual_tactical_note(
                    meta, info, events, xg_data, stats
                )
                _render_visual_page(
                    pdf,
                    figs[meta["idx"] - 1],
                    info,
                    meta,
                    statline,
                    commentary,
                    page_num,
                    total_pages,
                )
                page_num += 1

        # TACTICAL SUMMARY PAGE — deleted per user request
        d = pdf.infodict()
        _h_sc, _a_sc = _parse_scoreline(info, xg_data, events=events)
        d["Title"] = f"Tactical Report: {hn} {_h_sc}-{_a_sc} {an}"
        d["Author"] = "Mostafa Saad"
        d["Subject"] = f"{info.get('competition', '')} - {info.get('date', '')}"
        d["Keywords"] = "football tactical report, match analysis, PDF"

    console.print(
        f"\n[bold green]  Tactical PDF saved -> {pdf_path}[/bold green]\n"
        f"  [dim]{len(ordered_catalog)} visual pages grouped into shared, home and away sections[/dim]"
    )
    return pdf_path


# ══════════════════════════════════════════════════════════════════════
#  PDF BUILDER  (white portrait report style)
#  Inspired by the user's sample: cover -> executive summary -> visual + text pages
# ══════════════════════════════════════════════════════════════════════
PDF_PAGE_SIZE = (8.27, 11.69)  # A4 portrait in inches
PDF_EXPLANATION_STYLE = (
    "reference"  # "reference" = sample PDF text layout; "side_panel" = legacy note card
)
# ----------------------------------------------------------------------
# PDF dark theme: keeps the dark script identity while using the new
# reference-style explanation layout.
# ----------------------------------------------------------------------
PDF_BG = "#000000"
PDF_SURFACE = "#050505"
PDF_BORDER = "#233244"
PDF_TEXT = "#FFFFFF"
PDF_TEXT_DIM = "#9CA3AF"
PDF_ACCENT = "#FFD000"

# Compatibility aliases used by the existing PDF builder.
PDF_WHITE = PDF_BG
PDF_INK = PDF_TEXT
PDF_MUTED = PDF_TEXT_DIM
PDF_RULE = PDF_BORDER
PDF_GOLD_LINE = PDF_ACCENT


def _pdf_score_title(info, stats=None, events=None):
    hn, an = info["home_name"], info["away_name"]
    h_sc, a_sc = _parse_scoreline(info, {}, events=events)
    return f"{hn} {h_sc}-{a_sc} {an}"


def _pdf_header_line(info, events=None):
    bits = [
        _pdf_score_title(info, events=events),
        info.get("competition", ""),
        info.get("venue", ""),
    ]
    return " | ".join([b for b in bits if b])


def _pdf_draw_header_footer(fig, info, page_num, total_pages, events=None):
    """Header & footer for every PDF page in white background style."""
    fig.patch.set_facecolor(PDF_BG)

    header_ax = fig.add_axes([0.0, 0.955, 1.0, 0.045], zorder=10)
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.axis("off")
    header_ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, facecolor=PDF_SURFACE, edgecolor="none")
    )

    hn = info.get("home_name", "Home")
    header_ax.text(
        0.025,
        0.5,
        hn,
        ha="left",
        va="center",
        color=_pdf_team_text_color(HOME_COLOR),
        fontsize=11,
        fontweight="bold",
    )

    header_ax.text(
        0.5,
        0.5,
        _pdf_header_line(info, events=events),
        ha="center",
        va="center",
        color=PDF_TEXT,
        fontsize=10,
    )

    an = info.get("away_name", "Away")
    header_ax.text(
        0.975,
        0.5,
        an,
        ha="right",
        va="center",
        color=_pdf_team_text_color(AWAY_COLOR),
        fontsize=11,
        fontweight="bold",
    )

    fig.add_artist(
        plt.Line2D(
            [0.0, 0.5],
            [0.954, 0.954],
            transform=fig.transFigure,
            color=HOME_COLOR,
            lw=1.4,
        )
    )
    fig.add_artist(
        plt.Line2D(
            [0.5, 1.0],
            [0.954, 0.954],
            transform=fig.transFigure,
            color=AWAY_COLOR,
            lw=1.4,
        )
    )

    # ── Footer ──
    fig.add_artist(
        plt.Line2D(
            [0.055, 0.945],
            [0.040, 0.040],
            transform=fig.transFigure,
            color=PDF_BORDER,
            lw=0.6,
        )
    )
    fig.text(
        0.055,
        0.024,
        "Analysis by Mostafa Saad",
        ha="left",
        va="bottom",
        color=PDF_TEXT_DIM,
        fontsize=8,
    )
    fig.text(
        0.945,
        0.024,
        f"Page {page_num} of {total_pages}",
        ha="right",
        va="bottom",
        color=PDF_TEXT_DIM,
        fontsize=8,
    )


def _pdf_team_text_color(team_color):
    """
    Choose a readable team-name color on a white background.
    Use sufficiently dark team colors directly and replace very light colors
    with dark text.
    """
    if _is_light_color(team_color):
        return PDF_TEXT  # dark text for light team colors on white bg
    return team_color


def _blend_hex_with_white(color: str, amount: float = 0.82) -> str:
    """
    Return a soft tint of a team colour for PDF tables.
    Blend the team color with white to create a subtle table-cell tint.
    """
    r, g, b = _hex_to_rgb01(color)
    sr, sg, sb = _hex_to_rgb01("#FFFFFF")  # blend toward white
    amount = _clamp(amount, 0.0, 1.0)

    rr = int(round((r * (1 - amount) + sr * amount) * 255))
    gg = int(round((g * (1 - amount) + sg * amount) * 255))
    bb = int(round((b * (1 - amount) + sb * amount) * 255))
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def _pdf_write_wrapped(
    ax, text, x=0.0, y=1.0, width=96, fontsize=10.5, line_spacing=1.28, color=PDF_INK
):
    import textwrap

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cursor = y
    for para in str(text).split("\n"):
        para = para.strip()
        if not para:
            cursor -= 0.045
            continue
        wrapped = textwrap.wrap(para, width=width) or [""]
        for line in wrapped:
            ax.text(
                x,
                cursor,
                line,
                ha="left",
                va="top",
                color=color,
                fontsize=fontsize,
                family="serif",
            )
            cursor -= 0.024 * line_spacing
        cursor -= 0.020
    return cursor


def _pdf_section_heading(fig, section_title, subsection_title, accent="#38BDF8"):
    fig.text(
        0.055,
        0.925,
        section_title,
        ha="left",
        va="top",
        color=PDF_INK,
        fontsize=16,
        fontweight="bold",
        family="serif",
    )
    fig.add_artist(
        plt.Line2D(
            [0.055, 0.945],
            [0.900, 0.900],
            transform=fig.transFigure,
            color=PDF_GOLD_LINE,
            lw=0.75,
        )
    )
    fig.text(
        0.055,
        0.875,
        subsection_title,
        ha="left",
        va="top",
        color=accent,
        fontsize=13.5,
        fontweight="bold",
        family="serif",
    )


def _pdf_scorers_line(events, info):
    if events is None or events.empty or "is_goal" not in events.columns:
        return ""
    g = events[events["is_goal"] == True].copy()
    if g.empty:
        return ""
    items = []
    for _, row in g.sort_values(["minute", "second"]).iterrows():
        minute = int(_safe_float(row.get("minute"), 0))
        scored_for = (
            info.get("home_name")
            if row.get("scoring_team") == info.get("home_id")
            else info.get("away_name")
        )
        scorer = str(row.get("player") or "Unknown")
        og = " OG" if bool(row.get("is_own_goal", False)) else ""
        items.append(f"{minute}' {_short(scorer)} ({scored_for}{og})")
    return " | ".join(items)


def _render_cover_page(pdf, info, stats, events, total_pages):
    hn, an = info["home_name"], info["away_name"]
    h_sc, a_sc = _parse_scoreline(info, {}, events=events)
    h_color = HOME_COLOR
    a_color = AWAY_COLOR

    cover = plt.figure(figsize=PDF_PAGE_SIZE, facecolor=PDF_BG)
    ax = cover.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PDF_BG)

    ax.add_patch(
        plt.Rectangle((0.0, 0.965), 0.5, 0.035, facecolor=h_color, edgecolor="none")
    )
    ax.add_patch(
        plt.Rectangle((0.5, 0.965), 0.5, 0.035, facecolor=a_color, edgecolor="none")
    )

    ax.add_patch(
        plt.Rectangle(
            (0.055, 0.90),
            0.89,
            0.006,
            facecolor=PDF_ACCENT,
            edgecolor="none",
            alpha=0.95,
        )
    )

    ax.text(
        0.5,
        0.79,
        "MATCH ANALYSIS REPORT",
        ha="center",
        va="center",
        color=PDF_TEXT,
        fontsize=22,
        fontweight="bold",
    )

    score_y = 0.70

    cover.text(
        0.30,
        score_y,
        hn,
        ha="right",
        va="center",
        color=_pdf_team_text_color(h_color),
        fontsize=22,
        fontweight="bold",
        transform=cover.transFigure,
    )
    cover.text(
        0.50,
        score_y,
        f"{h_sc}  -  {a_sc}",
        ha="center",
        va="center",
        color=PDF_TEXT,
        fontsize=32,
        fontweight="bold",
        transform=cover.transFigure,
    )
    cover.text(
        0.70,
        score_y,
        an,
        ha="left",
        va="center",
        color=_pdf_team_text_color(a_color),
        fontsize=22,
        fontweight="bold",
        transform=cover.transFigure,
    )

    meta = " | ".join(
        [
            x
            for x in [
                info.get("competition", ""),
                info.get("venue", ""),
                info.get("date", ""),
            ]
            if x
        ]
    )
    ax.text(0.5, 0.625, meta, ha="center", va="center", color=PDF_TEXT_DIM, fontsize=12)

    scorers = _pdf_scorers_line(events, info)
    if scorers:
        ax.text(
            0.5,
            0.555,
            "Scorers",
            ha="center",
            va="center",
            color=PDF_ACCENT,
            fontsize=11.5,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.515,
            scorers,
            ha="center",
            va="center",
            color=PDF_TEXT,
            fontsize=9.4,
            wrap=True,
        )

    ax.add_patch(
        plt.Rectangle((0.20, 0.45), 0.60, 0.001, facecolor=PDF_BORDER, edgecolor="none")
    )

    ax.text(
        0.5,
        0.40,
        "Data: WhoScored | xG: Internal V7 event-context/team-stat model | xT: Karun Singh",
        ha="center",
        va="center",
        color=PDF_TEXT_DIM,
        fontsize=10,
    )
    ax.text(
        0.5,
        0.34,
        "Visuals & Analysis: Mostafa Saad",
        ha="center",
        va="center",
        color=PDF_TEXT,
        fontsize=12,
        fontweight="bold",
    )

    # ── footer ──
    ax.text(
        0.5,
        0.06,
        f"Analysis by Mostafa Saad  |  Page 1 of {total_pages}",
        ha="center",
        va="center",
        color=PDF_TEXT_DIM,
        fontsize=8,
    )

    pdf.savefig(
        cover,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_BG,
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(cover)


def _pdf_metric_table(ax, rows, hn, an, h_color, a_color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(rows) + 1
    row_h = 1.0 / n
    x0, w0, w1, w2 = 0.00, 0.48, 0.26, 0.26
    # Header row: use team colors + dark header for metric column
    ax.add_patch(
        plt.Rectangle(
            (x0, 1 - row_h), w0, row_h, facecolor="#1F2937", edgecolor=PDF_RULE, lw=0.6
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (x0 + w0, 1 - row_h),
            w1,
            row_h,
            facecolor=h_color,
            edgecolor=PDF_RULE,
            lw=0.6,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (x0 + w0 + w1, 1 - row_h),
            w2,
            row_h,
            facecolor=a_color,
            edgecolor=PDF_RULE,
            lw=0.6,
        )
    )
    ax.text(
        x0 + 0.02,
        1 - row_h / 2,
        "Metric",
        va="center",
        ha="left",
        color="#FFFFFF",
        fontsize=10,
        fontweight="bold",
        family="serif",
    )
    ax.text(
        x0 + w0 + w1 / 2,
        1 - row_h / 2,
        hn,
        va="center",
        ha="center",
        color=_text_on_color(h_color),
        fontsize=10,
        fontweight="bold",
        family="serif",
    )
    ax.text(
        x0 + w0 + w1 + w2 / 2,
        1 - row_h / 2,
        an,
        va="center",
        ha="center",
        color=_text_on_color(a_color),
        fontsize=10,
        fontweight="bold",
        family="serif",
    )
    for i, (metric, hv, av) in enumerate(rows):
        y = 1 - row_h * (i + 2)

        fill = PDF_SURFACE if i % 2 == 0 else PDF_BG
        ax.add_patch(
            plt.Rectangle(
                (x0, y), w0, row_h, facecolor=fill, edgecolor=PDF_RULE, lw=0.45
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x0 + w0, y),
                w1,
                row_h,
                facecolor=_blend_hex_with_white(h_color, 0.82),
                edgecolor=PDF_RULE,
                lw=0.45,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x0 + w0 + w1, y),
                w2,
                row_h,
                facecolor=_blend_hex_with_white(a_color, 0.82),
                edgecolor=PDF_RULE,
                lw=0.45,
            )
        )
        ax.text(
            x0 + 0.02,
            y + row_h / 2,
            metric,
            va="center",
            ha="left",
            color=PDF_INK,
            fontsize=9.5,
            fontweight="bold",
            family="serif",
        )
        ax.text(
            x0 + w0 + w1 / 2,
            y + row_h / 2,
            str(hv),
            va="center",
            ha="center",
            color=_pdf_team_text_color(h_color),
            fontsize=9.5,
            fontweight="bold",
            family="serif",
        )
        ax.text(
            x0 + w0 + w1 + w2 / 2,
            y + row_h / 2,
            str(av),
            va="center",
            ha="center",
            color=_pdf_team_text_color(a_color),
            fontsize=9.5,
            fontweight="bold",
            family="serif",
        )


def _render_executive_summary_page(
    pdf, info, stats, events, xg_data, page_num, total_pages
):
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    h_sc, a_sc = _parse_scoreline(info, xg_data, events=events)
    hxT = _fmt_num(_xt_total(events, info["home_id"]), 2)
    axT = _fmt_num(_xt_total(events, info["away_id"]), 2)
    page = plt.figure(figsize=PDF_PAGE_SIZE, facecolor=PDF_WHITE)
    _pdf_draw_header_footer(page, info, page_num, total_pages)
    page.text(
        0.055,
        0.925,
        "Executive Summary",
        ha="left",
        va="top",
        color=PDF_INK,
        fontsize=18,
        fontweight="bold",
        family="serif",
    )
    page.add_artist(
        plt.Line2D(
            [0.055, 0.945],
            [0.895, 0.895],
            transform=page.transFigure,
            color=PDF_GOLD_LINE,
            lw=0.75,
        )
    )
    try:
        winner = (
            hn
            if int(str(h_sc).strip() or 0) > int(str(a_sc).strip() or 0)
            else (
                an
                if int(str(a_sc).strip() or 0) > int(str(h_sc).strip() or 0)
                else "neither side"
            )
        )
    except Exception:
        winner = "the winning side"
    summary = (
        f"{hn} and {an} produced a {h_sc}-{a_sc} match shaped by chance quality, territory, pressing and ball progression. "
        f"The headline numbers show {hn} with {h['shots']} shots and {h['xG']} xG, while {an} recorded {a['shots']} shots and {a['xG']} xG. "
        f"The result went to {winner}, but the tactical reading is not only about the scoreline: it is about which side reached the best zones, how often those zones were accessed, and whether the final action matched the quality of the build-up.\n\n"
        f"In possession, {hn} completed passes at {h['pass_accuracy']}% accuracy compared with {an}'s {a['pass_accuracy']}%. "
        f"The xT totals - {hn}: {hxT}, {an}: {axT} - indicate how much of that possession became forward threat rather than simple circulation. "
        f"Progressive passes ({h['prog_passes']} vs {a['prog_passes']}) are important because they separate safe ball retention from genuine line-breaking actions that move the opposition block toward its own goal.\n\n"
        f"The report treats every visual as a tactical evidence point. Shot maps explain access to the box; xG and xGoT explain chance quality and finishing; pass networks show the build-up structure; Zone 14, box entries and crossing maps show the final-third route; and defensive/pressing charts explain how each side tried to control the match without the ball. "
        f"Read together, these sections provide a coaching-style interpretation of the match rather than a simple statistical recap."
    )
    text_ax = page.add_axes([0.055, 0.505, 0.89, 0.36])
    _pdf_write_wrapped(text_ax, summary, width=96, fontsize=10, line_spacing=1.18)
    rows = [
        ("Goals", h_sc, a_sc),
        ("xG", _fmt_num(h["xG"], 2), _fmt_num(a["xG"], 2)),
        ("xT", hxT, axT),
        (
            "Shots (On Target)",
            f"{h['shots']} ({h['on_target']})",
            f"{a['shots']} ({a['on_target']})",
        ),
        ("Pass Accuracy", f"{h['pass_accuracy']}%", f"{a['pass_accuracy']}%"),
        ("Progressive Passes", h["prog_passes"], a["prog_passes"]),
        ("Crosses", h["crosses_total"], a["crosses_total"]),
        ("High Turnovers", h["high_turnovers"], a["high_turnovers"]),
        (
            "GK Saves",
            h.get("saved", xg_data.get(hn, {}).get("saved", 0)),
            a.get("saved", xg_data.get(an, {}).get("saved", 0)),
        ),
    ]
    tbl_ax = page.add_axes([0.055, 0.165, 0.80, 0.28])
    _pdf_metric_table(tbl_ax, rows, hn, an, C_RED, C_BLUE)
    pdf.savefig(
        page,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_WHITE,
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(page)


def _pdf_section_for_meta(meta, info):
    hn, an = info["home_name"], info["away_name"]
    kind = meta.get("kind", "")
    if kind in {
        "shared_match_stats",
        "shared_xg_flow",
        "shared_shot_breakdown",
        "shared_shot_comparison",
        "shared_xg_tiles",
        "shared_gk_saves",
    }:
        return "1. Match Overview", "#38BDF8"
    if meta.get("team") == "home" and kind in {
        "team_shot_map",
        "team_danger_creation",
        "team_zone14",
        "team_box_entries",
        "team_crosses",
    }:
        return f"2. {hn} - Attacking Analysis", _pdf_team_text_color(C_RED)
    if meta.get("team") == "away" and kind in {
        "team_shot_map",
        "team_danger_creation",
        "team_zone14",
        "team_box_entries",
        "team_crosses",
    }:
        return f"3. {an} - Attacking Analysis", _pdf_team_text_color(C_BLUE)
    if meta.get("team") == "home" and kind in {
        "team_pass_network",
        "team_pass_thirds",
        "team_progressive_passes",
        "team_xt_map",
        "team_pass_target_zones",
    }:
        return f"4. {hn} - Build-up & Passing", _pdf_team_text_color(C_RED)
    if meta.get("team") == "away" and kind in {
        "team_pass_network",
        "team_pass_thirds",
        "team_progressive_passes",
        "team_xt_map",
        "team_pass_target_zones",
    }:
        return f"5. {an} - Build-up & Passing", _pdf_team_text_color(C_BLUE)
    if kind in {
        "shared_territorial",
        "shared_touches",
        "shared_dominating_zone",
        "team_average_positions",
    }:
        return "6. Territorial Control & Shape", "#38BDF8"
    if kind in {
        "team_high_turnovers",
        "team_def_heatmap",
        "shared_def_summary",
        "shared_xt_per_minute",
    }:
        return "7. Pressing & Defensive Work", "#38BDF8"
    return "8. Additional Match Visuals", "#38BDF8"


def _report_catalog_order(info):
    catalog = _build_visual_catalog(info)
    order = [
        16,
        1,
        4,
        9,
        13,
        2,
        10,
        14,
        32,
        24,
        3,
        11,
        15,
        33,
        25,
        5,
        19,
        22,
        7,
        36,
        6,
        20,
        23,
        8,
        37,
        17,
        18,
        31,
        29,
        30,
        34,
        35,
        26,
        27,
        28,
        12,
        21,
    ]
    rank = {idx: i for i, idx in enumerate(order)}
    return sorted(catalog, key=lambda m: rank.get(m["idx"], 999 + m["idx"]))


def _render_visual_page_side_panel(
    pdf, src_fig, info, meta, statline, commentary, page_num, total_pages, events=None
):
    section_title, accent = _pdf_section_for_meta(meta, info)
    page = plt.figure(figsize=PDF_PAGE_SIZE, facecolor=PDF_WHITE)
    _pdf_draw_header_footer(page, info, page_num, total_pages, events=events)
    _pdf_section_heading(page, section_title, meta["title"], accent=accent)

    img = _figure_to_rgba(src_fig)
    img_ax = page.add_axes([0.055, 0.405, 0.58, 0.445])
    img_ax.set_facecolor(PDF_SURFACE)
    img_ax.imshow(img, interpolation="lanczos")
    img_ax.axis("off")
    for spine in img_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PDF_RULE)
        spine.set_linewidth(0.6)

    panel_ax = page.add_axes([0.665, 0.405, 0.28, 0.445])
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, 1)
    panel_ax.axis("off")
    panel_ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor=PDF_SURFACE,
            edgecolor=PDF_RULE,
            lw=0.8,
            alpha=0.98,
        )
    )
    panel_ax.text(
        0.06,
        0.94,
        "Tactical Note",
        ha="left",
        va="top",
        color=accent,
        fontsize=10.5,
        fontweight="bold",
        family="serif",
    )
    panel_ax.text(
        0.06,
        0.86,
        _wrap_panel_text(statline, width=34),
        ha="left",
        va="top",
        color=PDF_MUTED,
        fontsize=8.6,
        family="serif",
    )
    panel_ax.plot([0.06, 0.94], [0.78, 0.78], color=PDF_GOLD_LINE, lw=0.7)
    panel_ax.text(
        0.06,
        0.74,
        _wrap_panel_text(commentary, width=34),
        ha="left",
        va="top",
        color=PDF_INK,
        fontsize=9.2,
        family="serif",
    )

    text_ax = page.add_axes([0.055, 0.065, 0.89, 0.285])
    body = f"{statline}\n\n{commentary}"
    _pdf_write_wrapped(text_ax, body, width=96, fontsize=9.4, line_spacing=1.15)
    pdf.savefig(
        page,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_WHITE,
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(page)


def _render_visual_page_reference(
    pdf, src_fig, info, meta, statline, commentary, page_num, total_pages, events=None
):
    section_title, accent = _pdf_section_for_meta(meta, info)
    page = plt.figure(figsize=PDF_PAGE_SIZE, facecolor=PDF_WHITE)
    _pdf_draw_header_footer(page, info, page_num, total_pages, events=events)
    _pdf_section_heading(page, section_title, meta["title"], accent=accent)

    img = _figure_to_rgba(src_fig)
    img_ax = page.add_axes([0.055, 0.525, 0.89, 0.325])
    img_ax.set_facecolor(PDF_SURFACE)
    img_ax.imshow(img, interpolation="lanczos")
    img_ax.axis("off")
    for spine in img_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PDF_RULE)
        spine.set_linewidth(0.6)

    text_ax = page.add_axes([0.055, 0.065, 0.89, 0.415])
    body = f"{statline}\n\n{commentary}"
    _pdf_write_wrapped(text_ax, body, width=96, fontsize=10, line_spacing=1.18)
    pdf.savefig(
        page,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_WHITE,
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(page)


def _render_visual_page(
    pdf, src_fig, info, meta, statline, commentary, page_num, total_pages, events=None
):
    style = str(globals().get("PDF_EXPLANATION_STYLE", "reference")).strip().lower()
    if style in {"side_panel", "legacy", "current"}:
        return _render_visual_page_side_panel(
            pdf,
            src_fig,
            info,
            meta,
            statline,
            commentary,
            page_num,
            total_pages,
            events=events,
        )
    return _render_visual_page_reference(
        pdf,
        src_fig,
        info,
        meta,
        statline,
        commentary,
        page_num,
        total_pages,
        events=events,
    )


def build_tactical_pdf(figs, info, events, xg_data, ts):
    """Assemble the final tactical PDF in Dark Mode with Cairo font and 300 DPI."""

    from matplotlib import font_manager as fm

    def _ensure_cairo_font():
        """
        Ensure that the Cairo font is available by searching common system
        paths, downloading it from Google Fonts when necessary, and registering
        it with matplotlib.font_manager. Return True on success and False when
        downloading fails, allowing the caller to use a fallback font.
        """

        system_paths = [
            r"C:\Windows\Fonts\Cairo-Regular.ttf",
            r"C:\Windows\Fonts\Cairo-Bold.ttf",
            r"C:\Windows\Fonts\Cairo-SemiBold.ttf",
            os.path.expanduser(
                "~/AppData/Local/Microsoft/Windows/Fonts/Cairo-Regular.ttf"
            ),
            os.path.expanduser(
                "~/AppData/Local/Microsoft/Windows/Fonts/Cairo-Bold.ttf"
            ),
            "/usr/share/fonts/truetype/cairo/Cairo-Regular.ttf",
            "/usr/share/fonts/truetype/cairo/Cairo-Bold.ttf",
            os.path.expanduser("~/Library/Fonts/Cairo-Regular.ttf"),
        ]
        found_any = False
        for p in system_paths:
            if os.path.exists(p):
                try:
                    fm.fontManager.addfont(p)
                    found_any = True
                except Exception:
                    pass
        if found_any:
            return True

        try:
            local_fonts_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "fonts"
            )
        except NameError:
            local_fonts_dir = os.path.join(os.getcwd(), "fonts")
        os.makedirs(local_fonts_dir, exist_ok=True)

        cairo_sources = {
            "Cairo-Regular.ttf": [
                "https://cdn.jsdelivr.net/npm/@fontsource/cairo@5.0.13/files/cairo-arabic-400-normal.ttf",
                "https://unpkg.com/@fontsource/cairo@5.0.13/files/cairo-arabic-400-normal.ttf",
                "https://cdn.jsdelivr.net/npm/@fontsource/cairo@5.0.13/files/cairo-latin-400-normal.ttf",
            ],
            "Cairo-Bold.ttf": [
                "https://cdn.jsdelivr.net/npm/@fontsource/cairo@5.0.13/files/cairo-arabic-700-normal.ttf",
                "https://unpkg.com/@fontsource/cairo@5.0.13/files/cairo-arabic-700-normal.ttf",
                "https://cdn.jsdelivr.net/npm/@fontsource/cairo@5.0.13/files/cairo-latin-700-normal.ttf",
            ],
        }

        downloaded = False
        for fname, urls in cairo_sources.items():
            local_path = os.path.join(local_fonts_dir, fname)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:

                try:
                    fm.fontManager.addfont(local_path)
                    found_any = True
                except Exception:
                    pass
                continue

            for url in urls:
                try:
                    import urllib.request

                    console.print(f"[dim]  Downloading Cairo font: {fname}[/dim]")
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = resp.read()
                        if len(data) < 1000:
                            continue
                        with open(local_path, "wb") as f:
                            f.write(data)
                        downloaded = True
                        break
                except Exception:
                    continue

            if os.path.exists(local_path):
                try:
                    fm.fontManager.addfont(local_path)
                    found_any = True
                except Exception:
                    pass

        if downloaded:

            try:
                fm._load_fontmanager(try_read_cache=False)
            except Exception:
                pass

        return found_any

    cairo_available = _ensure_cairo_font()
    if cairo_available:
        primary_font = "Cairo"
        console.print("[dim]  ✓ Cairo font registered[/dim]")
    else:
        primary_font = "DejaVu Sans"
        console.print(
            "[yellow]  ⚠ Cairo font not available — using DejaVu Sans fallback.[/yellow]\n"
            "[dim]    To install Cairo: download from https://fonts.google.com/specimen/Cairo[/dim]"
        )

    matplotlib.rcParams.update(
        {
            "figure.facecolor": PDF_BG,
            "axes.facecolor": PDF_SURFACE,
            "axes.edgecolor": PDF_BORDER,
            "axes.labelcolor": PDF_TEXT,
            "axes.titlecolor": PDF_TEXT,
            "xtick.color": PDF_TEXT_DIM,
            "ytick.color": PDF_TEXT_DIM,
            "text.color": PDF_TEXT,
            "grid.color": PDF_BORDER,
            "grid.alpha": 0.5,
            "savefig.facecolor": PDF_BG,
            "savefig.edgecolor": "none",
            "savefig.dpi": OUTPUT_IMAGE_DPI,
            "figure.dpi": 300,
            "font.family": [primary_font, "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )

    hn, an = info["home_name"], info["away_name"]
    stats = _ensure_match_stats_defaults(_collect_match_stats(info, events, xg_data))

    safe_hn = hn.replace(" ", "_").replace("/", "_")
    safe_an = an.replace(" ", "_").replace("/", "_")
    pdf_path = f"{SAVE_DIR}/match_analysis_report_{safe_hn}_vs_{safe_an}_{ts}.pdf"

    console.print(
        "\n[bold cyan]  Writing Dark Mode match analysis PDF report...[/bold cyan]"
    )
    console.print(f"[bold cyan]  Building PDF: {pdf_path}[/bold cyan]")

    ordered_catalog = [m for m in _report_catalog_order(info) if m["idx"] <= len(figs)]
    total_pages = 2 + len(ordered_catalog)
    page_num = 1

    with PdfPages(pdf_path) as pdf:
        _render_cover_page(pdf, info, stats, events, total_pages)
        page_num += 1
        _render_executive_summary_page(
            pdf, info, stats, events, xg_data, page_num, total_pages
        )
        page_num += 1

        for meta in ordered_catalog:
            statline, commentary = _visual_tactical_note(
                meta, info, events, xg_data, stats
            )
            _render_visual_page(
                pdf,
                figs[meta["idx"] - 1],
                info,
                meta,
                statline,
                commentary,
                page_num,
                total_pages,
                events=events,
            )
            page_num += 1

        d = pdf.infodict()
        _h_sc, _a_sc = _parse_scoreline(info, xg_data, events=events)
        d["Title"] = f"Match Analysis Report: {hn} {_h_sc}-{_a_sc} {an}"
        d["Author"] = "Mostafa Saad"
        d["Subject"] = f"{info.get('competition', '')} - {info.get('date', '')}"
        d["Keywords"] = "football match analysis, tactical report, WhoScored, xG, xT"

    console.print(
        f"\n[bold green]  Match analysis PDF saved -> {pdf_path}[/bold green]\n"
    )
    return pdf_path


# ══════════════════════════════════════════════════════
#  TERMINAL SUMMARY
# ══════════════════════════════════════════════════════
def print_summary(info, xg_data, events):
    console.rule(
        f"[bold cyan]  {info['home_name']}  {info['score']}  "
        f"{info['away_name']}  [/bold cyan]"
    )
    console.print(
        f"  Venue: {info['venue']}   |   "
        f"Formations: {info['home_form']} vs {info['away_form']}",
        justify="center",
    )
    xt = Table(
        title="Shot Breakdown",
        header_style="bold magenta",
        show_lines=True,
        border_style="dim",
    )
    for col, style in [
        ("Team", "cyan"),
        ("xG", "green"),
        ("Shots", ""),
        ("On Target", "green"),
        ("Goals", "yellow"),
        ("Saves", "green"),
        ("Off Target", "red"),
        ("Blocked", "orange3"),
        ("Woodwork", "blue"),
        ("Big Ch.", ""),
    ]:
        xt.add_column(
            col, style=style, justify="center", min_width=16 if col == "Team" else 7
        )
    for name, s in xg_data.items():
        xt.add_row(
            name,
            str(s["xG"]),
            str(s["shots"]),
            str(s["on_target"]),
            str(s["goals"]),
            str(s["saved"]),
            str(s["missed"]),
            str(s["blocked"]),
            str(s["post"]),
            str(s["big_chances"]),
        )
    console.print(xt)

    pss = events[events["is_pass"] == True]
    if not pss.empty:
        pt = Table(
            title="Pass Stats",
            header_style="bold blue",
            show_lines=True,
            border_style="dim",
        )
        for col, style, just in [
            ("Team", "cyan", "left"),
            ("Total", "", "center"),
            ("Completed", "green", "center"),
            ("Accuracy", "green", "center"),
            ("Key Passes", "yellow", "center"),
        ]:
            pt.add_column(
                col, style=style, justify=just, min_width=16 if col == "Team" else 8
            )
        for side in ["home", "away"]:
            tid = info[f"{side}_id"]
            name = info[f"{side}_name"]
            tp = pss[pss["team_id"] == tid]
            tot = len(tp)
            suc = int((tp["outcome"] == "Successful").sum())
            acc = round(suc / tot * 100, 1) if tot else 0
            key = int(tp["is_key_pass"].sum())
            pt.add_row(name, str(tot), str(suc), f"{acc}%", str(key))
        console.print(pt)

    gdf = events[events["is_goal"] == True]
    if not gdf.empty:
        gt = Table(
            title="Goals",
            header_style="bold yellow",
            show_lines=True,
            border_style="dim",
        )
        gt.add_column("Min", justify="center", width=5)
        gt.add_column("Scorer", style="bold white", min_width=18)
        gt.add_column("Scored For", style="cyan", min_width=14)
        gt.add_column("Type", justify="center", width=24)
        gt.add_column("Assist", style="green", min_width=18)
        gt.add_column("xG", justify="center", style="yellow", width=6)
        for _, row in gdf.iterrows():
            scored_for = (
                info["home_name"]
                if row["scoring_team"] == info["home_id"]
                else info["away_name"]
            )
            from match_report import (
                classify_goal_type as _classify_goal_type,
                goal_body_part_label as _goal_body_part_label,
            )

            if row.get("is_own_goal", False):
                goal_type = "[bold magenta]OWN GOAL[/bold magenta]"
            else:
                cat, sub = _classify_goal_type(row, events)
                body = _goal_body_part_label(row)
                goal_type = (
                    f"{cat} - {body}"
                    if cat != "Set Piece"
                    else f"Set Piece - {sub} - {body}"
                )
            gt.add_row(
                f"{row['minute']}'",
                _short(str(row["player"])),
                scored_for,
                goal_type,
                _short(str(row["assist_player"])) if row["assist_player"] else "—",
                f"{row['xG']:.3f}" if row["xG"] else "—",
            )
        console.print(gt)


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    global SAVE_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)

    md = scrape_match(
        MATCH_URL,
        chromedriver_path=CHROMEDRIVER_PATH,
        profile_dir=CHROME_PROFILE_DIR,
        profile_name=CHROME_PROFILE_NAME,
    )

    info, events, players = parse_all(md)
    SAVE_DIR = _match_output_folder(info)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.environ["MATCH_ANALYSIS_OUTPUT_DIR"] = SAVE_DIR
    console.print(f"[cyan]  Output folder -> {SAVE_DIR}[/cyan]")

    global HOME_COLOR, AWAY_COLOR, C_RED, C_BLUE
    home_col, away_col = choose_matchup_colors(
        info.get("home_name", ""),
        info.get("away_name", ""),
        HOME_KIT_TYPE,
        AWAY_KIT_TYPE,
    )

    HOME_COLOR = home_col
    AWAY_COLOR = away_col
    C_RED = home_col
    C_BLUE = away_col
    info["home_color"] = home_col
    info["away_color"] = away_col
    info["home_kit_type"] = HOME_KIT_TYPE
    info["away_kit_type"] = AWAY_KIT_TYPE

    console.print(
        f"[dim]  Team colors: {info.get('home_name', '?')} ({HOME_KIT_TYPE}) = {home_col}  |  "
        f"{info.get('away_name', '?')} ({AWAY_KIT_TYPE}) = {away_col}[/dim]"
    )

    # First try the official team stats already embedded in matchCentreData.
    # This is the most stable path and avoids Selenium completely when available.
    mc_stats = _extract_matchcentre_stats(md)
    if _official_stats_has(mc_stats):
        page_stats = mc_stats
        console.print(
            "[green]  Using matchCentreData stat counts; xG totals will be calculated by the internal V7 model (browser skipped).[/green]"
        )
    else:
        console.print(
            "[yellow]  matchCentreData stat counts incomplete; trying HTTP/HTML stats capture before any browser...[/yellow]"
        )
        try:
            dom_stats = _get_official_stats(
                info,
                MATCH_URL,
                chromedriver_path=CHROMEDRIVER_PATH,
                profile_dir=CHROME_PROFILE_DIR,
                profile_name=CHROME_PROFILE_NAME,
            )
            page_stats = _merge_official_stats(mc_stats, dom_stats)
        except Exception as _off_err:
            console.print(
                f"[yellow]  ⚠ Official Opta stats fetch failed: {_off_err}[/yellow]\n"
                f"[yellow]  → Official counts will be kept; xG will be calculated by the internal V7 model.[/yellow]"
            )
            page_stats = (
                mc_stats  # may be empty/partial — the local model will fill the gaps
            )

    # V7: keep official/matchCentre counts, but remove any provider/public team xG total.
    # The xG total is produced internally from the event-level model and available team stats.
    page_stats = _strip_external_xg_totals(page_stats)
    info["xg_reference_source"] = "v7 internal event/team-stat model"

    info["official_stats"] = _finalize_official_stats(page_stats)
    info["official_stats"] = _fill_missing_xg_with_public_fallback(info, events)
    if STRICT_OFFICIAL_PAGE_XG:
        missing_xg = [
            side
            for side in ("home", "away")
            if info.get("official_stats", {}).get(side, {}).get("xG") is None
        ]
        if missing_xg:
            raise RuntimeError(
                "Internal V7 xG was not produced for both teams. "
                "This strict version will not output fallback xG totals. "
                f"Missing: {', '.join(missing_xg)}. "
                "Try opening the match page manually in Chrome first, then re-run."
            )
    events = _apply_official_stats_calibration(info, events)
    xg_data = xg_stats(events, info)
    status = get_status(md)
    if info.get("official_stats"):
        console.print(
            f"[green]  Using V7 internal xG model for report totals. Stat counts source: matchCentreData/DOM when available. Model: {XG_MODEL_USED}.[/green]"
        )
        console.print(info["official_stats"])
    else:
        console.print(
            f"[yellow]  Official stat counts not found; using event-derived counts and internal V7 xG.[/yellow]"
        )
    sub_in = info["sub_in"]
    sub_out = info["sub_out"]
    red_cards = info["red_cards"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SAVE_DIR, exist_ok=True)
    events.to_csv(
        os.path.join(SAVE_DIR, "events.csv"), index=False, encoding="utf-8-sig"
    )
    players.to_csv(
        os.path.join(SAVE_DIR, "players.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(xg_data).T.reset_index().rename(columns={"index": "team"}).to_csv(
        os.path.join(SAVE_DIR, "xg.csv"), index=False, encoding="utf-8-sig"
    )

    print_summary(info, xg_data, events)

    plt.style.use("dark_background")
    figs = []
    figs_filenames = []  # parallel list — used by the PDF builder to group
    # figures into Home / Away / Shared sections by
    # inspecting "_home_" / "_away_" in the filename

    # ── shorthand — available to ALL helpers and figure calls below ───
    hn, an = info["home_name"], info["away_name"]
    hid, aid = info["home_id"], info["away_id"]

    def _fig(w, h, title=""):
        f = plt.figure(figsize=(w, h), facecolor=BG_DARK)
        if title and SHOW_WINDOWS and getattr(f.canvas, "manager", None):
            try:
                f.canvas.manager.set_window_title(title)
            except Exception:
                pass
        return f

    # ── shared header helpers (must be defined before any fig call) ────────
    def _add_header(fig, title, subtitle=""):
        """
        Colour bar (home | away), stat name, key numbers.
        Credit "Created by Mostafa Saad" in CENTRE of the colour bar.
        """
        # ── colour bar ────────────────────────────────────────────
        cax = fig.add_axes([0.0, 0.981, 1.0, 0.019])
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.axis("off")
        cax.add_patch(
            plt.Rectangle((0, 0), 0.50, 1, facecolor=C_RED, alpha=0.92, zorder=0)
        )
        cax.add_patch(
            plt.Rectangle((0.50, 0), 0.50, 1, facecolor=C_BLUE, alpha=0.92, zorder=0)
        )
        cax.plot(
            [0.50, 0.50], [0.08, 0.92], color="white", lw=0.8, alpha=0.35, zorder=2
        )
        # Home name — left
        cax.text(
            0.015,
            0.50,
            f"● {hn[:20]}",
            ha="left",
            va="center",
            color=_text_on_color(C_RED),
            fontsize=7.5,
            fontweight="bold",
            zorder=3,
        )
        # Credit — CENTRE of bar
        cax.text(
            0.50,
            0.50,
            "Created by Mostafa Saad",
            ha="center",
            va="center",
            color="#FFD700",
            fontsize=7.8,
            fontweight="bold",
            fontstyle="italic",
            zorder=3,
        )
        # Away name — right
        cax.text(
            0.985,
            0.50,
            f"{an[:20]} ●",
            ha="right",
            va="center",
            color=_text_on_color(C_BLUE),
            fontsize=7.5,
            fontweight="bold",
            zorder=3,
        )
        # ── Line 1: stat name (with glow) ─────────────────────────
        fig.text(
            0.50,
            0.966,
            title,
            ha="center",
            va="top",
            color=TEXT_BRIGHT,
            fontsize=15,
            fontweight="bold",
            transform=fig.transFigure,
            path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
        )
        # ── Line 2: key numbers ───────────────────────────────────
        if subtitle:
            fig.text(
                0.50,
                0.928,
                subtitle,
                ha="center",
                va="top",
                color=TEXT_DIM,
                fontsize=9,
                transform=fig.transFigure,
            )

    # ══════════════════════════════════════════════════════
    #  Visuals 1..8 — REDESIGNED v2 (xG flow, shot maps, shot breakdown,
    #  pass networks, xT maps). All built from the unified design system
    #  (chrome + panel cards + key insight + bottom metric strip).
    #  Falls back to the legacy renderers if tactical_visualizations is unavailable.
    # ══════════════════════════════════════════════════════
    def _save_and_append(fig, fname):
        """Save the v2 figure (its chrome already includes the unified
        identity, so we DON'T add the legacy _watermark on top — it would
        duplicate the yellow bars + footer)."""
        fname = _clean_output_filename(fname)
        _apply_neon_backdrop(fig)
        # On a RAM-tight machine the Agg renderer for a 420-DPI save can fail
        # with MemoryError. Rather than abort the whole run (losing the PDF and
        # every later fig), free memory and retry the save at progressively
        # lower DPI so the fig still lands — just a touch softer.
        for _dpi in (OUTPUT_IMAGE_DPI, 300, 200):
            try:
                fig.savefig(
                    f"{SAVE_DIR}/{fname}",
                    dpi=_dpi,
                    bbox_inches="tight",
                    facecolor=BG_DARK,
                )
                break
            except Exception as _sv_err:
                import gc as _gc

                _gc.collect()
                if _dpi == 200:
                    console.print(
                        f"[yellow]  ⚠ Could not save {fname}: {_sv_err}[/yellow]"
                    )
                elif _dpi == OUTPUT_IMAGE_DPI:
                    console.print(
                        f"[yellow]  ⚠ {fname} save hit memory pressure — "
                        f"retrying at lower DPI[/yellow]"
                    )
        figs.append(fig)
        figs_filenames.append(fname)

    def _clean_output_filename(fname):
        root, ext = os.path.splitext(str(fname))
        root = re.sub(r"_\d{8}_\d{6}(?=$|_)", "", root)
        return f"{root}{ext}"

    if _V2_AVAILABLE:
        # 1. xG Flow
        fig1 = make_xg_flow_v2(events, info, xg_data)
        _save_and_append(fig1, f"1_xg_flow_{ts}.png")
        # 2. Shot Map Home
        fig2 = make_shot_map_v2(events, info, info["home_id"], C_RED)
        _save_and_append(fig2, f"2_shot_map_home_{ts}.png")
        # 3. Shot Map Away
        fig3 = make_shot_map_v2(events, info, info["away_id"], C_BLUE)
        _save_and_append(fig3, f"3_shot_map_away_{ts}.png")
        # 4. Shot Breakdown & Goals
        fig4 = make_shot_breakdown_v2(events, info, xg_data)
        _save_and_append(fig4, f"4_breakdown_goals_{ts}.png")
        # 5. Pass Network Home
        fig5 = make_pass_network_v2(events, info, info["home_id"], C_RED)
        _save_and_append(fig5, f"5_pass_network_home_{ts}.png")
        # 6. Pass Network Away
        fig6 = make_pass_network_v2(events, info, info["away_id"], C_BLUE)
        _save_and_append(fig6, f"6_pass_network_away_{ts}.png")
        # 7. xT Map Home
        fig7 = make_xt_map_v2(events, info, info["home_id"], C_RED)
        _save_and_append(fig7, f"7_xt_map_home_{ts}.png")
        # 8. xT Map Away
        fig8 = make_xt_map_v2(events, info, info["away_id"], C_BLUE)
        _save_and_append(fig8, f"8_xt_map_away_{ts}.png")
    else:
        # ── Legacy fallback (only runs if tactical_visualizations failed to import) ──
        fig1 = _fig(15, 7, "Fig 1 — xG Flow")
        ax1 = fig1.add_subplot(
            GridSpec(1, 1, figure=fig1, left=0.07, right=0.97, top=0.88, bottom=0.11)[
                0, 0
            ]
        )
        _add_header(
            fig1,
            "xG Flow",
            f"{hn}: xG {xg_data.get(hn,{}).get('xG',0):.2f}  |  "
            f"{an}: xG {xg_data.get(an,{}).get('xG',0):.2f}",
        )
        draw_xg_flow(fig1, ax1, events, info, xg_data, status)
        _watermark(fig1)
        _save_and_append(fig1, f"1_xg_flow_{ts}.png")

        fig2 = _fig(14, 12, f"Fig 2 — Shot Map: {info['home_name']}")
        draw_shot_map_full(fig2, events, info["home_id"], info["home_name"], C_RED)
        _watermark(fig2)
        _save_and_append(fig2, f"2_shot_map_home_{ts}.png")

        fig3 = _fig(14, 12, f"Fig 3 — Shot Map: {info['away_name']}")
        draw_shot_map_full(fig3, events, info["away_id"], info["away_name"], C_BLUE)
        _watermark(fig3)
        _save_and_append(fig3, f"3_shot_map_away_{ts}.png")

        fig4 = _fig(16, 13, "Fig 4 — Breakdown & Goals")
        _add_header(
            fig4,
            "Shot Breakdown & Goals",
            f"{hn}: {xg_data.get(hn,{}).get('shots',0)} shots  "
            f"xG {xg_data.get(hn,{}).get('xG',0):.2f}   |   "
            f"{an}: {xg_data.get(an,{}).get('shots',0)} shots  "
            f"xG {xg_data.get(an,{}).get('xG',0):.2f}",
        )
        draw_breakdown_goals(fig4, events, info, xg_data)
        _watermark(fig4)
        _save_and_append(fig4, f"4_breakdown_goals_{ts}.png")

        fig5 = _fig(18, 11, f"Fig 5 — Pass Network: {info['home_name']}")
        draw_pass_network_full(
            fig5,
            events,
            info["home_id"],
            info["home_name"],
            C_RED,
            sub_in,
            sub_out,
            red_cards,
        )
        _watermark(fig5)
        _save_and_append(fig5, f"5_pass_network_home_{ts}.png")

        fig6 = _fig(18, 11, f"Fig 6 — Pass Network: {info['away_name']}")
        draw_pass_network_full(
            fig6,
            events,
            info["away_id"],
            info["away_name"],
            C_BLUE,
            sub_in,
            sub_out,
            red_cards,
        )
        _watermark(fig6)
        _save_and_append(fig6, f"6_pass_network_away_{ts}.png")

        fig7 = _fig(18, 11, f"Fig 7 — xT Map: {info['home_name']}")
        draw_xt_map_full(fig7, events, info["home_id"], info["home_name"], C_RED)
        _watermark(fig7)
        _save_and_append(fig7, f"7_xt_map_home_{ts}.png")

        fig8 = _fig(18, 11, f"Fig 8 — xT Map: {info['away_name']}")
        draw_xt_map_full(fig8, events, info["away_id"], info["away_name"], C_BLUE)
        _watermark(fig8)
        _save_and_append(fig8, f"8_xt_map_away_{ts}.png")

    # ══════════════════════════════════════════════════════
    #  STANDALONE VISUALS — each stat in its own figure
    # ══════════════════════════════════════════════════════

    def _sf(w, h, label, subtitle="", team_color=None, team_name=None):
        """
        Standalone figure — enhanced header.

        team_color + team_name → single-team figure:
          Full-width team-color bar  |  left: team name  |  centre: credit
        No team args → both-teams figure:
          Split bar home/away  |  left: home  |  centre: credit  |  right: away

        Line 1 (large, bold, white+glow) : stat name
        Line 2 (smaller, dim)            : key numbers
        """
        f = plt.figure(figsize=(w, h), facecolor=BG_DARK)

        # ── colour bar ────────────────────────────────────────────────
        cax = f.add_axes([0.0, 0.980, 1.0, 0.020])
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.axis("off")

        if team_color and team_name:
            # Single team: full-width band in team colour
            cax.add_patch(
                plt.Rectangle(
                    (0, 0), 1.0, 1, facecolor=team_color, alpha=0.93, zorder=0
                )
            )
            # Subtle highlight strip at top edge
            cax.add_patch(
                plt.Rectangle(
                    (0, 0.82), 1.0, 0.18, facecolor="white", alpha=0.07, zorder=1
                )
            )
            # Team name — left
            cax.text(
                0.015,
                0.50,
                f"● {team_name}",
                ha="left",
                va="center",
                color=_text_on_color(team_color),
                fontsize=8.5,
                fontweight="bold",
                zorder=3,
            )
            # Credit — centre
            cax.text(
                0.50,
                0.50,
                "Created by Mostafa Saad",
                ha="center",
                va="center",
                color=_accent_on_color(team_color),
                fontsize=8,
                fontweight="bold",
                fontstyle="italic",
                zorder=3,
            )
        else:
            # Both teams: split band
            cax.add_patch(
                plt.Rectangle((0, 0), 0.50, 1, facecolor=C_RED, alpha=0.91, zorder=0)
            )
            cax.add_patch(
                plt.Rectangle(
                    (0.50, 0), 0.50, 1, facecolor=C_BLUE, alpha=0.91, zorder=0
                )
            )
            # Thin white separator at centre
            cax.plot(
                [0.50, 0.50], [0.08, 0.92], color="white", lw=0.8, alpha=0.35, zorder=2
            )
            # Home name — left
            cax.text(
                0.015,
                0.50,
                f"● {hn[:18]}",
                ha="left",
                va="center",
                color=_text_on_color(C_RED),
                fontsize=8,
                fontweight="bold",
                zorder=3,
            )
            # Credit — centre
            cax.text(
                0.50,
                0.50,
                "Created by Mostafa Saad",
                ha="center",
                va="center",
                color="#FFD700",
                fontsize=7.8,
                fontweight="bold",
                fontstyle="italic",
                zorder=3,
            )
            # Away name — right
            cax.text(
                0.985,
                0.50,
                f"{an[:18]} ●",
                ha="right",
                va="center",
                color=_text_on_color(C_BLUE),
                fontsize=8,
                fontweight="bold",
                zorder=3,
            )

        # ── Line 1: stat name (with glow) ─────────────────────────────
        f.text(
            0.50,
            0.962,
            label,
            ha="center",
            va="top",
            color=TEXT_BRIGHT,
            fontsize=15,
            fontweight="bold",
            transform=f.transFigure,
            path_effects=[pe.withStroke(linewidth=3, foreground="#000000")],
        )

        # ── Line 2: key numbers ───────────────────────────────────────
        if subtitle:
            f.text(
                0.50,
                0.928,
                subtitle,
                ha="center",
                va="top",
                color=TEXT_DIM,
                fontsize=9,
                transform=f.transFigure,
            )

        return f

    def _sp(fig, lp=0.06, rp=0.95, tp=0.82, bp=0.10):
        """Single subplot below the three-line header (title + subtitle + credit).
        top=0.82  → clears header area
        bottom=0.10 → room for axis labels + watermark
        """
        return fig.add_subplot(
            GridSpec(1, 1, figure=fig, left=lp, right=rp, top=tp, bottom=bp)[0, 0]
        )

    def _sv(fig, fname):
        """Watermark + save."""
        fname = os.path.join(
            os.path.dirname(fname), _clean_output_filename(os.path.basename(fname))
        )
        _watermark(fig)
        fig.savefig(
            fname,
            dpi=PDF_EXPORT_DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            pad_inches=0.1,
        )
        figs.append(fig)
        # Track only the basename (without dir path) so the PDF builder
        # can group by filename pattern.
        figs_filenames.append(os.path.basename(fname))

    base = f"{SAVE_DIR}"

    # ── pre-compute subtitle data ─────────────────────────────────
    _hxg = xg_data.get(hn, {})
    _axg = xg_data.get(an, {})
    _h_xg = _hxg.get("xG", 0)
    _a_xg = _axg.get("xG", 0)
    _h_xgt = _hxg.get("xGoT", 0)
    _a_xgt = _axg.get("xGoT", 0)
    _h_sh = _hxg.get("shots", 0)
    _a_sh = _axg.get("shots", 0)
    _h_ot = _hxg.get("on_target", 0)
    _a_ot = _axg.get("on_target", 0)
    _h_sv = _hxg.get("saved", 0)
    _a_sv = _axg.get("saved", 0)

    # ── 11–22: figs A→L wired to v2 redesigns ───────────────────
    if _V2_AVAILABLE:
        from tactical_visualizations import (
            make_shot_comparison_v2,
            make_danger_creation_v2,
            make_gk_saves_v2,
            make_xg_summary_v2,
            make_zone14_v2,
            make_match_stats_v2,
            make_territorial_v2,
            make_ball_touches_v2,
            make_pass_thirds_v2,
        )

        # Compute PPDA on the fly (match_stats_v2 needs it)
        try:
            from match_report import compute_ppda_both

            _ppda = compute_ppda_both(info, events)
        except Exception:
            _ppda = {"home": {}, "away": {}}

        fa = make_shot_comparison_v2(events, info, xg_data)
        _save_and_append(fa, f"11_shot_comparison_{ts}.png")
        fb = make_danger_creation_v2(events, info, hid, C_RED)
        _save_and_append(fb, f"12_danger_home_{ts}.png")
        fc = make_danger_creation_v2(events, info, aid, C_BLUE)
        _save_and_append(fc, f"13_danger_away_{ts}.png")
        # GK Saves — keep the legacy chart, wrap in v2 chrome + sidebar/strip
        from tactical_visualizations import render_legacy_chart_v2
        from visualization_components import C_GOLD as _GOLD_V2

        # Build sidebar/strip data from events
        _h_faced = events[(events["team_id"] == aid) & (events["is_shot"] == True)]
        _a_faced = events[(events["team_id"] == hid) & (events["is_shot"] == True)]
        _h_xg_faced = float(_h_faced["xG"].fillna(0).sum() if not _h_faced.empty else 0)
        _a_xg_faced = float(_a_faced["xG"].fillna(0).sum() if not _a_faced.empty else 0)
        _h_saves = int(
            (_h_faced.get("shot_whoscored_type") == "SavedShot").sum()
            if "shot_whoscored_type" in _h_faced.columns
            else (_h_faced["type"] == "SavedShot").sum()
        )
        _a_saves = int(
            (_a_faced.get("shot_whoscored_type") == "SavedShot").sum()
            if "shot_whoscored_type" in _a_faced.columns
            else (_a_faced["type"] == "SavedShot").sum()
        )
        _h_g = int(_h_faced["is_goal"].sum() if not _h_faced.empty else 0)
        _a_g = int(_a_faced["is_goal"].sum() if not _a_faced.empty else 0)

        fd = render_legacy_chart_v2(
            section="GOALKEEPER SAVES",
            title=f"{hn} vs {an} — Keeper Saves",
            subtitle="Each dot = a shot the keeper faced · size = xG · "
            "filled = goal · ringed = saved",
            hn=hn,
            an=an,
            score=str(info.get("score") or "—"),
            footer_note="Save quality scales with the xG of shots faced",
            team_color=_GOLD_V2,
            draw_legacy=lambda fig, ax: _panel_gk_saves(ax, events, info),
            sidebar_title="Keepers head-to-head",
            sidebar_headers=["KEEPER", "FACED", "XG"],
            sidebar_rows=[
                (f"{hn[:14]} GK", str(len(_h_faced)), f"{_h_xg_faced:.2f}"),
                (f"{an[:14]} GK", str(len(_a_faced)), f"{_a_xg_faced:.2f}"),
                ("Saves", f"{_h_saves} / {_a_saves}", "—"),
                ("Goals conceded", f"{_h_g} / {_a_g}", "—"),
            ],
            insight_text=(
                f"{hn}'s keeper faced {len(_h_faced)} shots ({_h_xg_faced:.2f} xG) "
                f"and made {_h_saves} saves. {an}'s keeper faced "
                f"{len(_a_faced)} ({_a_xg_faced:.2f} xG), {_a_saves} saves."
            ),
            metric_cards=[
                (f"{hn[:14]} xG faced", f"{_h_xg_faced:.2f}", C_RED),
                (f"{hn[:14]} Saves", str(_h_saves), _GOLD_V2),
                ("Total Shots", str(len(_h_faced) + len(_a_faced)), _GOLD_V2),
                (f"{an[:14]} Saves", str(_a_saves), _GOLD_V2),
                (f"{an[:14]} xG faced", f"{_a_xg_faced:.2f}", C_BLUE),
            ],
        )
        _save_and_append(fd, f"14_gk_saves_{ts}.png")
        fe = make_xg_summary_v2(events, info, xg_data)
        _save_and_append(fe, f"15_xg_tiles_{ts}.png")
        ff = make_zone14_v2(events, info, hid, C_RED)
        _save_and_append(ff, f"16_zone14_home_{ts}.png")
        fg = make_zone14_v2(events, info, aid, C_BLUE)
        _save_and_append(fg, f"17_zone14_away_{ts}.png")
        fh = make_match_stats_v2(events, info, _ppda)
        _save_and_append(fh, f"18_match_stats_{ts}.png")
        # fig 19 (Territorial Control) intentionally removed — it was an
        # exact duplicate of fig 33 (Dominating Zone), so we keep only
        # the Dominating Zone framing later in the sequence.
        # Ball Touches — legacy donut + v2 chrome/sidebar/strip
        from tactical_visualizations import render_legacy_chart_v2 as _legacy_v2
        from visualization_components import C_GOLD as _GOLD_V2

        _h_touches = int(((events["team_id"] == hid) & events["x"].notna()).sum())
        _a_touches = int(((events["team_id"] == aid) & events["x"].notna()).sum())
        _diff = _h_touches - _a_touches
        _leader = hn if _diff > 0 else an

        # Per-third diff
        def _zone_diff(x_lo, x_hi):
            h = int(
                ((events["team_id"] == hid) & events["x"].between(x_lo, x_hi)).sum()
            )
            a = int(
                ((events["team_id"] == aid) & events["x"].between(x_lo, x_hi)).sum()
            )
            return h - a

        _att = _zone_diff(67, 100)
        _mid = _zone_diff(33, 67)
        _def = _zone_diff(0, 33)

        fj = _legacy_v2(
            section="BALL TOUCHES",
            title=f"{hn} vs {an} — Ball Touches",
            subtitle="Donut shows each side's share of touches per pitch "
            "third · whoever owned the territory owned the game",
            hn=hn,
            an=an,
            score=str(info.get("score") or "—"),
            footer_note="Where the game actually got played",
            team_color=_GOLD_V2,
            draw_legacy=lambda fig, ax: _panel_donut_dual(ax, events, info),
            sidebar_title="Touch Distribution (H − A)",
            sidebar_headers=["WHERE", "DIFF"],
            sidebar_rows=[
                ("Att third", f"{_att:+d}"),
                ("Mid third", f"{_mid:+d}"),
                ("Def third", f"{_def:+d}"),
                (f"{hn[:14]}", str(_h_touches)),
                (f"{an[:14]}", str(_a_touches)),
            ],
            insight_text=(
                f"{_leader} touched the ball {abs(_diff)} more times "
                f"overall. The donut breaks the share down by pitch third "
                f"so you can see WHERE each side dominated."
            ),
            metric_cards=[
                (f"{hn[:14]} Touches", str(_h_touches), C_RED),
                ("Total", str(_h_touches + _a_touches), _GOLD_V2),
                (f"{an[:14]} Touches", str(_a_touches), C_BLUE),
                ("Diff", f"{'+' if _diff >= 0 else ''}{_diff}", _GOLD_V2),
                ("Leader", _leader[:10], _GOLD_V2),
            ],
        )
        _save_and_append(fj, f"20_possession_{ts}.png")
        fk = make_pass_thirds_v2(events, info, hid, C_RED)
        _save_and_append(fk, f"21_pass_thirds_home_{ts}.png")
        fl = make_pass_thirds_v2(events, info, aid, C_BLUE)
        _save_and_append(fl, f"22_pass_thirds_away_{ts}.png")
    else:
        fa = _sf(
            13,
            5.0,
            "Shot Comparison",
            subtitle=f"{hn}  vs  {an}   |   xG: {_h_xg:.2f} – {_a_xg:.2f}   |   Shots: {_h_sh} – {_a_sh}",
        )
        _panel_shot_comparison(_sp(fa), events, info, xg_data)
        _sv(fa, f"{base}/11_shot_comparison_{ts}.png")
        fb = _sf(
            10,
            8,
            "Danger Creation",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}   |   Shots: {_h_sh}   On Target: {_h_ot}   xG: {_h_xg:.2f}",
        )
        _panel_danger(_sp(fb, lp=0.02, rp=0.98), events, hid, C_RED, hn)
        _sv(fb, f"{base}/12_danger_home_{ts}.png")
        fc = _sf(
            10,
            8,
            "Danger Creation",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}   |   Shots: {_a_sh}   On Target: {_a_ot}   xG: {_a_xg:.2f}",
        )
        _panel_danger(_sp(fc, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
        _sv(fc, f"{base}/13_danger_away_{ts}.png")
        fd = _sf(
            10,
            6,
            "Goalkeeper Saves",
            subtitle=f"{hn}: {_h_sv} saves   |   {an}: {_a_sv} saves",
        )
        _panel_gk_saves(_sp(fd), events, info)
        _sv(fd, f"{base}/14_gk_saves_{ts}.png")
        fe = _sf(
            13,
            5.0,
            "xG / xGoT / On Target",
            subtitle=f"{hn}  xG {_h_xg:.2f}   xGoT {_h_xgt:.2f}   |   {an}  xG {_a_xg:.2f}   xGoT {_a_xgt:.2f}",
        )
        _panel_xg_tiles(_sp(fe), events, info, xg_data)
        _sv(fe, f"{base}/15_xg_tiles_{ts}.png")
        ff = _sf(
            10,
            8,
            "Zone 14 & Half-Spaces",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}   |   Actions in Zone 14 and Half-Space channels",
        )
        _panel_zone14(_sp(ff, lp=0.02, rp=0.98), events, hid, C_RED, hn)
        _sv(ff, f"{base}/16_zone14_home_{ts}.png")
        fg = _sf(
            10,
            8,
            "Zone 14 & Half-Spaces",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}   |   Actions in Zone 14 and Half-Space channels",
        )
        _panel_zone14(_sp(fg, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
        _sv(fg, f"{base}/17_zone14_away_{ts}.png")
        fh = _sf(
            9, 10, "Match Statistics", subtitle=f"{hn}  vs  {an}   |   {info['venue']}"
        )
        _panel_match_stats(_sp(fh, lp=0.07, rp=0.93), events, info, xg_data)
        _sv(fh, f"{base}/18_match_stats_{ts}.png")
        # fig 19 (Territorial Control) intentionally removed — it was an
        # exact duplicate of fig 33 (Dominating Zone).
        fj = _sf(
            9,
            6,
            "Ball Touches",
            subtitle=f"{hn}  vs  {an}   |   Touch distribution by zone",
        )
        _panel_donut_dual(_sp(fj), events, info)
        _sv(fj, f"{base}/20_possession_{ts}.png")
        fk = _sf(
            10,
            8,
            "Pass Map by Third",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}   |   Completed and incomplete passes across thirds",
        )
        _panel_pass_thirds(_sp(fk, lp=0.02, rp=0.98), events, hid, C_RED, hn)
        _sv(fk, f"{base}/21_pass_thirds_home_{ts}.png")
        fl = _sf(
            10,
            8,
            "Pass Map by Third",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}   |   Completed and incomplete passes across thirds",
        )
        _panel_pass_thirds(_sp(fl, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
        _sv(fl, f"{base}/22_pass_thirds_away_{ts}.png")

    # ── 23–27: xT per minute, progressives H/A, crosses H/A (v2) ─
    if _V2_AVAILABLE:
        from tactical_visualizations import (
            make_xt_per_minute_v2,
            make_progressive_passes_v2,
            make_crosses_v2,
        )

        fm = make_xt_per_minute_v2(events, info)
        _save_and_append(fm, f"23_xt_per_minute_{ts}.png")
        fn_ = make_progressive_passes_v2(events, info, hid, C_RED)
        _save_and_append(fn_, f"24_progressive_home_{ts}.png")
        fo = make_progressive_passes_v2(events, info, aid, C_BLUE)
        _save_and_append(fo, f"25_progressive_away_{ts}.png")
        fp1 = make_crosses_v2(events, info, hid, C_RED)
        _save_and_append(fp1, f"26_crosses_home_{ts}.png")
        fp2 = make_crosses_v2(events, info, aid, C_BLUE)
        _save_and_append(fp2, f"27_crosses_away_{ts}.png")
    else:
        fm = _sf(
            12,
            6,
            "xT per Minute",
            subtitle=f"{hn} (▲ red)  vs  {an} (▼ blue)   |   Expected Threat generated each minute",
        )
        _panel_xt_minute(_sp(fm, lp=0.08, rp=0.97, bp=0.11), events, info)
        _sv(fm, f"{base}/23_xt_per_minute_{ts}.png")
        fn_ = _sf(
            10,
            8,
            "Progressive Passes",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}   |   Passes moving the ball significantly toward goal",
        )
        _panel_progressive(_sp(fn_, lp=0.02, rp=0.98), events, hid, C_RED, hn)
        _sv(fn_, f"{base}/24_progressive_home_{ts}.png")
        fo = _sf(
            10,
            8,
            "Progressive Passes",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}   |   Passes moving the ball significantly toward goal",
        )
        _panel_progressive(_sp(fo, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
        _sv(fo, f"{base}/25_progressive_away_{ts}.png")
        fp1 = _sf(
            10,
            8,
            "Crosses",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}   |   Successful (solid) and unsuccessful (faded) crosses",
        )
        _panel_crosses_team(_sp(fp1, lp=0.02, rp=0.98), events, hid, C_RED, hn)
        _sv(fp1, f"{base}/26_crosses_home_{ts}.png")
        fp2 = _sf(
            10,
            8,
            "Crosses",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}   |   Successful (solid) and unsuccessful (faded) crosses",
        )
        _panel_crosses_team(_sp(fp2, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
        _sv(fp2, f"{base}/27_crosses_away_{ts}.png")

    # ── Q/R: Defensive Heatmap — Home + Away (v2) ────────────────
    if _V2_AVAILABLE:
        from tactical_visualizations import make_defensive_heatmap_v2

        fq = make_defensive_heatmap_v2(events, info, hid, C_RED)
        _save_and_append(fq, f"28_defensive_hm_home_{ts}.png")
        fr = make_defensive_heatmap_v2(events, info, aid, C_BLUE)
        _save_and_append(fr, f"29_defensive_hm_away_{ts}.png")
    else:
        fq = _sf(
            10,
            8,
            "Defensive Actions",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}   |   Tackles · Interceptions · Recoveries · Clearances · Aerials",
        )
        _panel_defensive_heatmap(
            _sp(fq, lp=0.02, rp=0.98), events, hid, C_RED, hn, info
        )
        _sv(fq, f"{base}/28_defensive_hm_home_{ts}.png")
        fr = _sf(
            10,
            8,
            "Defensive Actions",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}   |   Tackles · Interceptions · Recoveries · Clearances · Aerials",
        )
        _panel_defensive_heatmap(
            _sp(fr, lp=0.02, rp=0.98), events, aid, C_BLUE, an, info
        )
        _sv(fr, f"{base}/29_defensive_hm_away_{ts}.png")

    # ── 30: Defensive Summary (v2) ──────────────────────────────
    if _V2_AVAILABLE:
        from tactical_visualizations import make_defensive_summary_v2

        fs = make_defensive_summary_v2(events, info)
        _save_and_append(fs, f"30_defensive_summary_{ts}.png")
    else:
        fs = _sf(
            9,
            8,
            "Defensive Summary",
            subtitle=f"{hn}  vs  {an}   |   Count of each defensive action type",
        )
        _panel_def_counts(_sp(fs, lp=0.07, rp=0.93), events, info)
        _sv(fs, f"{base}/30_defensive_summary_{ts}.png")

    # ── T/U: Avg Positions — Home + Away (v2) ──────────────────────
    if _V2_AVAILABLE:
        from tactical_visualizations import make_avg_positions_v2

        ft = make_avg_positions_v2(events, info, hid, C_RED)
        _save_and_append(ft, f"31_avg_position_home_{ts}.png")
        fu = make_avg_positions_v2(events, info, aid, C_BLUE)
        _save_and_append(fu, f"32_avg_position_away_{ts}.png")
    else:
        ft = _sf(
            10,
            8,
            "Average Positions",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}   |   Mean touch position per player (size = touches)",
        )
        _panel_avg_position(_sp(ft, lp=0.02, rp=0.98), events, hid, C_RED, hn)
        _sv(ft, f"{base}/31_avg_position_home_{ts}.png")
        fu = _sf(
            10,
            8,
            "Average Positions",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}   |   Mean touch position per player (size = touches)",
        )
        _panel_avg_position(_sp(fu, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
        _sv(fu, f"{base}/32_avg_position_away_{ts}.png")

    # ══════════════════════════════════════════════════════
    #  NEW FIGURES 33–39
    # ══════════════════════════════════════════════════════

    # ── 33: Dominating Zone (v2) ────────────────────────
    if _V2_AVAILABLE:
        from tactical_visualizations import make_dominating_zone_v2

        f33 = make_dominating_zone_v2(events, info)
        _save_and_append(f33, f"33_dominating_zone_{ts}.png")
    else:
        f33 = _sf(
            14,
            8,
            "Dominating Zone",
            subtitle=f"{hn}  vs  {an}  |  >55% touches = dominant  |  45-55% = contested",
        )
        _panel_dominating_zone(
            _sp(f33, lp=0.03, rp=0.97, tp=0.84, bp=0.10), events, info
        )
        _sv(f33, f"{base}/33_dominating_zone_{ts}.png")

    # ── 34/35: Box Entries — Home + Away (v2) ────────────────────
    if _V2_AVAILABLE:
        from tactical_visualizations import make_box_entries_v2

        f34 = make_box_entries_v2(events, info, hid, C_RED)
        _save_and_append(f34, f"34_box_entries_home_{ts}.png")
        f35 = make_box_entries_v2(events, info, aid, C_BLUE)
        _save_and_append(f35, f"35_box_entries_away_{ts}.png")
    else:
        f34 = _sf(
            8,
            11,
            "Box Entries",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}  |  Passes & carries ending in opponent's penalty box",
        )
        _panel_box_entries(
            _sp(f34, lp=0.05, rp=0.95, tp=0.84, bp=0.06), events, hid, C_RED, hn
        )
        _sv(f34, f"{base}/34_box_entries_home_{ts}.png")
        f35 = _sf(
            8,
            11,
            "Box Entries",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}  |  Passes & carries ending in opponent's penalty box",
        )
        _panel_box_entries(
            _sp(f35, lp=0.05, rp=0.95, tp=0.84, bp=0.06), events, aid, C_BLUE, an
        )
        _sv(f35, f"{base}/35_box_entries_away_{ts}.png")

    # ── 36/37: High Turnovers — Home + Away (legacy identity, narrow pitch) ─
    if _V2_AVAILABLE:
        from tactical_visualizations import render_legacy_chart_v2 as _legacy_v2_ht
        from visualization_components import C_GOLD as _GOLD_HT

        REGAIN = {"Tackle", "Interception", "BallRecovery"}
        for _tid, _tc, _name, _opp_name, _idx, _suffix in [
            (hid, C_RED, hn, an, 36, "home"),
            (aid, C_BLUE, an, hn, 37, "away"),
        ]:
            _sub = events[
                (events["team_id"] == _tid)
                & events["type"].isin(list(REGAIN))
                & (events["x"] >= 60)
            ]
            _by_player = {}
            _types = {}
            for _, _r in _sub.iterrows():
                _p = str(_r.get("player") or "—")
                _by_player[_p] = _by_player.get(_p, 0) + 1
                _t = _r.get("type")
                _types[_t] = _types.get(_t, 0) + 1
            _top = sorted(_by_player.items(), key=lambda kv: -kv[1])[:6]
            _rows = [(p.split()[-1] if p else "—", str(c)) for p, c in _top]
            _leader = _top[0][0].split()[-1] if _top else "—"

            _fig_ht = _legacy_v2_ht(
                section="HIGH TURNOVERS",
                title=f"{_name} — High Turnovers",
                subtitle="Possession regains inside the final 40m — the "
                "press's tangible reward",
                hn=_name,
                an=_opp_name,
                score=str(info.get("score") or "—"),
                footer_note="High = inside the opposition half (x ≥ 60)",
                team_color=_tc,
                draw_legacy=(
                    lambda tid=_tid, tc=_tc, nm=_name: lambda fig, ax: _panel_high_turnovers(
                        ax, events, tid, tc, nm
                    )
                )(),
                sidebar_title="Top High-Pressers",
                sidebar_headers=["PLAYER", "REGAINS"],
                sidebar_rows=_rows,
                insight_text=(
                    f"{_name} regained possession {len(_sub)} times in the "
                    f"final 40 metres. {_leader} led with "
                    f"{_top[0][1] if _top else 0} high turnovers."
                ),
                metric_cards=[
                    ("High Regains", str(len(_sub)), _GOLD_HT),
                    ("Tackles", str(_types.get("Tackle", 0)), _tc),
                    ("Intercepts", str(_types.get("Interception", 0)), _GOLD_HT),
                    ("Recoveries", str(_types.get("BallRecovery", 0)), _tc),
                    ("Top Player", _leader, _GOLD_HT),
                ],
                legacy_box=[0.165, 0.145, 0.245, 0.705],
            )
            _save_and_append(_fig_ht, f"{_idx}_high_turnovers_{_suffix}_{ts}.png")
    else:
        f36 = _sf(
            8,
            11,
            "High Turnovers",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}  |  Ball wins within 40m of opponent goal",
        )
        _panel_high_turnovers(
            _sp(f36, lp=0.05, rp=0.95, tp=0.84, bp=0.06), events, hid, C_RED, hn
        )
        _sv(f36, f"{base}/36_high_turnovers_home_{ts}.png")
        f37 = _sf(
            8,
            11,
            "High Turnovers",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}  |  Ball wins within 40m of opponent goal",
        )
        _panel_high_turnovers(
            _sp(f37, lp=0.05, rp=0.95, tp=0.84, bp=0.06), events, aid, C_BLUE, an
        )
        _sv(f37, f"{base}/37_high_turnovers_away_{ts}.png")

    # ── 38/39: Pass Target Zones — Home + Away (legacy identity, narrow pitch) ─
    if _V2_AVAILABLE:
        from tactical_visualizations import render_legacy_chart_v2 as _legacy_v2_pt
        from visualization_components import C_GOLD as _GOLD_PT

        for _tid, _tc, _name, _opp_name, _idx, _suffix in [
            (hid, C_RED, hn, an, 38, "home"),
            (aid, C_BLUE, an, hn, 39, "away"),
        ]:
            _sub = events[
                (events["team_id"] == _tid)
                & (events["is_pass"] == True)
                & (events["outcome"] == "Successful")
                & events["end_x"].notna()
                & events["end_y"].notna()
            ]
            _n_total = len(_sub)
            _n_att = int((_sub["end_x"] >= 67).sum())
            _n_mid = int(_sub["end_x"].between(33, 67).sum())
            _n_def = int((_sub["end_x"] < 33).sum())
            _by_player = {}
            for _, _r in _sub.iterrows():
                _p = str(_r.get("player") or "—")
                _by_player[_p] = _by_player.get(_p, 0) + 1
            _top = sorted(_by_player.items(), key=lambda kv: -kv[1])[:6]

            _fig_pt = _legacy_v2_pt(
                section="PASS TARGET ZONES",
                title=f"{_name} — Pass Target Zones",
                subtitle="Where successful passes ended up · the team's "
                "preferred receiving real estate",
                hn=_name,
                an=_opp_name,
                score=str(info.get("score") or "—"),
                footer_note="Where the team wanted the ball to arrive",
                team_color=_tc,
                draw_legacy=(
                    lambda tid=_tid, tc=_tc, nm=_name: lambda fig, ax: _panel_pass_target_zones(
                        ax, events, tid, tc, nm
                    )
                )(),
                sidebar_title="Receiving by Third",
                sidebar_headers=["WHERE", "PASSES"],
                sidebar_rows=[
                    ("Att third", str(_n_att)),
                    ("Mid third", str(_n_mid)),
                    ("Def third", str(_n_def)),
                ]
                + [(p.split()[-1] if p else "—", str(c)) for p, c in _top[:2]],
                insight_text=(
                    f"{_name} found targets {_n_total} times. "
                    f"{(_n_att / _n_total * 100 if _n_total else 0):.0f}% "
                    f"of the team's successful passes landed in the "
                    f"attacking third."
                ),
                metric_cards=[
                    ("Targets", str(_n_total), _GOLD_PT),
                    ("Att 3rd", str(_n_att), _tc),
                    ("Mid 3rd", str(_n_mid), _GOLD_PT),
                    ("Def 3rd", str(_n_def), _tc),
                    (
                        "Att 3rd %",
                        f"{(_n_att / _n_total * 100 if _n_total else 0):.0f}%",
                        _GOLD_PT,
                    ),
                ],
                legacy_box=[0.165, 0.145, 0.245, 0.705],
            )
            _save_and_append(_fig_pt, f"{_idx}_pass_target_{_suffix}_{ts}.png")
    else:
        f38 = _sf(
            8,
            11,
            "Pass Target Zones",
            team_color=C_RED,
            team_name=hn,
            subtitle=f"{hn}  |  % of successful passes received per zone",
        )
        _panel_pass_target_zones(
            _sp(f38, lp=0.05, rp=0.95, tp=0.84, bp=0.06), events, hid, C_RED, hn
        )
        _sv(f38, f"{base}/38_pass_target_home_{ts}.png")
        f39 = _sf(
            8,
            11,
            "Pass Target Zones",
            team_color=C_BLUE,
            team_name=an,
            subtitle=f"{an}  |  % of successful passes received per zone",
        )
        _panel_pass_target_zones(
            _sp(f39, lp=0.05, rp=0.95, tp=0.84, bp=0.06), events, aid, C_BLUE, an
        )
        _sv(f39, f"{base}/39_pass_target_away_{ts}.png")

    # ── Shot Summary Tiles removed ───────────────────────────────
    # Removed by request: this visual rebuilds shot buckets from raw events and can
    # diverge from the official WhoScored/Opta totals. It is no longer saved and is
    # not included in the PDF report.

    # ══════════════════════════════════════════════════════
    #  Fig 40 — PPDA gauge (player-stat tables removed by request)
    # ══════════════════════════════════════════════════════
    try:
        from match_report import draw_ppda_gauge, compute_ppda_both

        # 40 — PPDA gauge
        try:
            _ppda_data = compute_ppda_both(info, events)
        except Exception:
            _ppda_data = {"home": {}, "away": {}}
        try:
            _f40 = draw_ppda_gauge(_ppda_data, info, save_path=None)
            _save_and_append(_f40, f"40_ppda_gauge_{ts}.png")
        except Exception as _e40:
            console.print(f"[yellow]  ⚠ PPDA fig (40) failed: {_e40}[/yellow]")
    except Exception as _ext_inline_err:
        console.print(f"[yellow]  ⚠ Inline PPDA failed: {_ext_inline_err}[/yellow]")

    # ══════════════════════════════════════════════════════
    #  CATEGORY SUMMARY BOARDS (4 grouped collages)
    # ══════════════════════════════════════════════════════
    board_paths = []
    try:
        board_paths = build_visual_category_boards(
            figs, info, events, xg_data, ts, figs_filenames=figs_filenames
        )
        if board_paths:
            console.print(
                f"[green]  Built {len(board_paths)} grouped summary boards.[/green]"
            )
    except Exception as _board_err:
        console.print(
            f"[yellow]  ⚠ Summary board generation failed: {_board_err}[/yellow]"
        )
        import traceback

        traceback.print_exc()

    # ══════════════════════════════════════════════════════
    #  Legacy tactical PDFs are intentionally skipped — the unified
    #  match_report_<ts>.pdf produced by the extension below is now the
    #  single, comprehensive output (covers + player ratings + goals log +
    #  team stats + PPDA + every tactical visual with its own
    #  'Reading this visual' commentary page).
    # ══════════════════════════════════════════════════════

    total_figs = 40  # Includes PPDA; player-stat tables removed by request
    extra_boards = len(board_paths)
    console.print(
        f"\n[bold green]  ✅ {total_figs} figures saved → {SAVE_DIR}/[/bold green]\n"
        f"  [dim]Figs  1-8  : individual analytics[/dim]\n"
        f"  [dim]Figs  9-32 : standalone visuals[/dim]\n"
        f"  [dim]Figs 33-40 : Dominating Zone · Box Entries · High Turnovers · Pass Target Zones · PPDA[/dim]\n"
        f"  [dim]{extra_boards} grouped summary boards added[/dim]\n"
        f"  [dim]Shot Summary Tiles and player-stat tables removed by request[/dim]"
    )

    # ── Extended pipeline (PPDA, goals log, unified PDF) ──
    try:
        # The unified report now embeds the 39 tactical figs directly via
        # extra_figs and adds a 'Reading this visual' commentary page after
        # each one — so we DON'T merge with the legacy tactical PDFs any more.
        # That keeps the output to a single, comprehensive match_report PDF.
        ext_result = _run_extended_analysis(
            md,
            parse_all_fn=parse_all,
            extra_figs=figs,
            extra_figs_filenames=figs_filenames,
            merge_with_pdfs=None,
        )
        console.print(f"[green]  ✓ Final unified PDF → {ext_result['pdf']}[/green]")
        # PPDA (40) is already produced inline as a numbered fig in SAVE_DIR.
    except Exception as _ext_err:
        import traceback as _tb

        console.print(
            f"[red]  ✗ Extended analysis FAILED — these outputs were NOT "
            f"generated: PPDA gauge, team-stats compare, "
            f"unified PDF.[/red]"
        )
        console.print(f"[red]    Error: {_ext_err}[/red]")
        console.print(f"[dim]{_tb.format_exc()}[/dim]")

    if SHOW_WINDOWS:
        plt.show()
    else:
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════
#  ENGLISH PDF OVERRIDE - full visual coverage + structured tactical notes
#  This block overrides the earlier Arabic PDF note builder at runtime.
# ══════════════════════════════════════════════════════════════════════


def _shared_section_summary(info, stats, events):
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    hg, ag = _parse_scoreline(
        info,
        {hn: {"goals": h.get("goals", 0)}, an: {"goals": a.get("goals", 0)}},
        events=events,
    )
    hxt = _fmt_num(_xt_total(events, info["home_id"]), 2)
    axt = _fmt_num(_xt_total(events, info["away_id"]), 2)
    leader_xg = _leader_name(h.get("xG", 0), a.get("xG", 0), hn, an)
    leader_prog = _leader_name(h.get("prog_passes", 0), a.get("prog_passes", 0), hn, an)
    return (
        f"{hn} vs {an} finished {hg}-{ag}. The report reads the score through chance quality, territory, pressing and ball progression.\n\n"
        f"Shots {h.get('shots', 0)}-{a.get('shots', 0)} | xG {h.get('xG', 0)}-{a.get('xG', 0)} | "
        f"On target {h.get('on_target', 0)}-{a.get('on_target', 0)} | Progressive passes {h.get('prog_passes', 0)}-{a.get('prog_passes', 0)} | xT {hxt}-{axt}\n\n"
        f"{leader_xg} produced the stronger chance-quality profile, while {leader_prog} created more forward-passing momentum. "
        f"The following pages explain the mechanisms behind those numbers."
    )


def _team_section_summary(side, info, stats, events):
    team = stats[side]
    other = "away" if side == "home" else "home"
    opp = stats[other]
    team_name = info[f"{side}_name"]
    opp_name = info[f"{other}_name"]
    team_xt = _xt_total(events, info[f"{side}_id"])
    opp_xt = _xt_total(events, info[f"{other}_id"])
    shots = team.get("shots", 0)
    avg_xg = round(team.get("xG", 0) / max(shots, 1), 2) if shots else 0.0
    return (
        f"{team_name} tactical profile against {opp_name}.\n\n"
        f"Shots {shots} | On target {team.get('on_target', 0)} | Goals {team.get('goals', 0)} | "
        f"xG {team.get('xG', 0)} ({avg_xg:.2f} per shot) | xT {_fmt_num(team_xt, 2)} | "
        f"Progressive passes {team.get('prog_passes', 0)} | Box entries {team.get('box_entries', 0)}\n\n"
        f"The main question is how efficiently {team_name} connected build-up, final-third access and defensive response. "
        f"The opponent comparison is xG {team.get('xG', 0)} vs {opp.get('xG', 0)} and xT {_fmt_num(team_xt, 2)} vs {_fmt_num(opp_xt, 2)}."
    )


def _next_visual_bridge(next_meta, info):
    """One linking sentence that hands the reader off to the next page, so the
    report reads as a connected tactical argument rather than isolated charts."""
    if not next_meta:
        return (
            "This is the final analytical page — read it back against the whole "
            "sequence above: chance quality, field position, ball progression and "
            "pressure after loss should all tell the same story."
        )
    nt = str(next_meta.get("title", "the next page"))
    nk = next_meta.get("kind", "")
    reason = {
        "shared_xg_flow": "to see exactly when these patterns moved the scoreline",
        "shared_shot_breakdown": "to separate this from raw shot volume and finishing variance",
        "shared_shot_comparison": "to weigh quantity against genuine chance quality",
        "shared_xg_tiles": "to compare chance quality before the shot with quality after contact",
        "shared_gk_saves": "to see how much real danger the goalkeeper actually faced",
        "shared_territorial": "to see where on the pitch this control was happening",
        "shared_touches": "to see whether possession was safe, central or threatening",
        "shared_dominating_zone": "to turn this control into pitch geography",
        "shared_xt_per_minute": "to follow the momentum of ball progression minute by minute",
        "shared_def_summary": "to see how the defensive workload was distributed",
        "team_shot_map": "to check whether this turned into clean shooting positions",
        "team_pass_network": "to see the passing structure underneath it",
        "team_xt_map": "to see which progression routes carried the threat",
        "team_box_entries": "to see how often the attack reached the penalty area",
        "team_zone14": "to see how the central connection zones were used",
        "team_danger_creation": "to see how build-up converted into end product",
        "team_crosses": "to judge the wide-delivery plan against box occupation",
        "team_progressive_passes": "to see how the ball was driven up the pitch",
        "team_pass_thirds": "to see where possession actually settled",
        "team_pass_target_zones": "to see where the next receiver was wanted",
        "team_average_positions": "to see the shape that underpinned all of this",
        "team_def_heatmap": "to see how the same side defended without the ball",
        "team_high_turnovers": "to see whether the press created attacking value",
    }.get(nk, "to add the next layer of the tactical picture")
    return f"Read next alongside “{nt}” {reason}."


def _visual_tactical_note(meta, info, events, xg_data, stats, next_meta=None):
    """English-only notes for every PDF visual."""
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    hid, aid = info["home_id"], info["away_id"]
    hg, ag = _parse_scoreline(info, xg_data, events=events)
    kind = meta.get("kind", "")

    def g(d, key, default=0):
        try:
            return d.get(key, default)
        except Exception:
            return default

    def fmt(v, digits=2):
        return _fmt_num(v, digits)

    if meta.get("team") == "home":
        side_key, other_key, team_name, opp_name, tid = "home", "away", hn, an, hid
    elif meta.get("team") == "away":
        side_key, other_key, team_name, opp_name, tid = "away", "home", an, hn, aid
    else:
        side_key = other_key = team_name = opp_name = tid = None

    if side_key:
        team = stats[side_key]
        opp = stats[other_key]
        team_xt = _xt_total(events, info[f"{side_key}_id"])
        opp_xt = _xt_total(events, info[f"{other_key}_id"])
        statline = (
            f"{team_name}: shots {g(team,'shots')} | xG {g(team,'xG')} | xT {fmt(team_xt)} | "
            f"progressive passes {g(team,'prog_passes')} | box entries {g(team,'box_entries')} | defensive actions {g(team,'defensive_acts')}"
        )
    else:
        team = opp = {}
        team_xt = opp_xt = 0
        hxT = _xt_total(events, hid)
        axT = _xt_total(events, aid)
        statline = (
            f"Score {hg}-{ag} | xG {g(h,'xG')}-{g(a,'xG')} | shots {g(h,'shots')}-{g(a,'shots')} | "
            f"on target {g(h,'on_target')}-{g(a,'on_target')} | xT {fmt(hxT)}-{fmt(axT)}"
        )

    shared_notes = {
        "shared_match_stats": "This page is the match baseline. Possession and passing only matter when they connect to xT, progressive passes, box entries and chance quality.",
        "shared_xg_flow": "The xG flow shows when the match tilted. Sharp jumps usually come from central shots, cut-backs, transition attacks or set pieces; flat periods mean possession existed without penetration.",
        "shared_shot_breakdown": "Shot volume and shot value are different. On-target shots show execution, blocked shots show defensive pressure, and xG shows whether the locations were genuinely valuable.",
        "shared_shot_comparison": "This comparison separates quantity from quality. If one side leads shots but not xG, it probably produced volume from lower-value areas.",
        "shared_xg_tiles": "xG measures chance quality before the shot; xGoT measures the quality of the shot after contact. The gap explains finishing value.",
        "shared_gk_saves": "Goalkeeper saves should be read with xGoT faced. High saves plus high xGoT faced means the goalkeeper protected the scoreline from real danger.",
        "shared_territorial": "Territorial control explains where the game lived. Attacking-third presence is more useful than raw possession because it keeps the opponent defending close to goal.",
        "shared_touches": "Touch distribution shows whether possession was safe, central or threatening. The attacking-third share is the key number for pressure.",
        "shared_dominating_zone": "Zone dominance translates possession into geography. It becomes valuable when it leads to box entries, cut-backs, shots or second-ball pressure.",
        "shared_xt_per_minute": "xT per minute is the momentum page for ball progression. Repeated spikes suggest a stable attacking route; isolated spikes suggest transitions or individual actions.",
        "shared_def_summary": "Defensive totals need context: tackles show duels, interceptions show anticipation, and recoveries show control of loose-ball moments.",
    }
    team_notes = {
        "team_shot_map": f"The shot map shows how {team_name} reached the final action. Central and close-range shots suggest clean penetration; wide or long-range shots suggest {opp_name} protected the middle.",
        "team_danger_creation": f"Danger creation links build-up to end product. For {team_name}, box entries show access, Zone 14 actions show central connection, and key passes show the final-ball quality.",
        "team_zone14": f"Zone 14 and the half-spaces are the main connection zones. Frequent actions there mean {team_name} accessed the space between {opp_name}'s midfield and defensive lines.",
        "team_box_entries": f"Box entries measure whether {team_name}'s attacks reached the most valuable area. The next action after the entry is the decisive tactical detail.",
        "team_crosses": f"Crossing volume only matters if {team_name} also occupied the box well. Low completion can mean poor delivery, but also underloaded penalty-area structure.",
        "team_pass_network": f"The pass network is the structure page for {team_name}. Dense links reveal circulation hubs; vertical links reveal routes through pressure.",
        "team_pass_thirds": f"The pass map by third shows where {team_name}'s possession settled: build-up under pressure, midfield control, or sustained final-third attacks.",
        "team_progressive_passes": f"Progressive passes show how {team_name} moved the opponent backward. The best ones break a line and give the receiver time to face forward.",
        "team_xt_map": f"Expected Threat values ball movement before the shot. {team_name}'s highest-xT zones identify the routes {opp_name} struggled to control.",
        "team_pass_target_zones": f"Pass target zones reveal intention: where {team_name} wanted the next receiver. Wide concentration suggests isolations; central concentration points to No. 10 or striker connections.",
        "team_average_positions": f"Average positions show occupation over time, not a fixed formation. The spacing explains compactness, counter-pressing potential and transition risk.",
        "team_def_heatmap": f"The defensive heatmap shows where {team_name} had to solve problems without the ball. High actions point to pressing; deep actions point to box protection.",
        "team_high_turnovers": f"High turnovers are the best measure of whether {team_name}'s press created attacking value. The key is whether regains quickly led to shots or box entries.",
    }
    base_note = shared_notes.get(kind) if not side_key else team_notes.get(kind)
    if not base_note:
        base_note = "This visual should be read as a tactical evidence page, connected to chance quality, field position, ball progression and pressing after loss."

    commentary = _expert_tactical_commentary(
        kind,
        base_note,
        meta,
        info,
        stats,
        events,
        xg_data,
        side_key=side_key,
        other_key=other_key,
        team=team if side_key else None,
        opp=opp if side_key else None,
        team_xt=team_xt,
        opp_xt=opp_xt,
    )
    commentary = commentary + "\n\n" + _next_visual_bridge(next_meta, info)
    return statline, commentary


def _render_board_image_page(
    pdf, image_path, info, title, page_num, total_pages, events=None
):
    page = plt.figure(figsize=PDF_PAGE_SIZE, facecolor=PDF_BG)
    _pdf_draw_header_footer(page, info, page_num, total_pages, events=events)
    _pdf_section_heading(page, "Grouped Summary Boards", title, accent=PDF_ACCENT)
    ax = page.add_axes([0.045, 0.075, 0.91, 0.79])
    ax.set_facecolor(PDF_SURFACE)
    try:
        img = plt.imread(image_path)
        ax.imshow(img, interpolation="lanczos")
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"Board image could not be loaded:\n{image_path}\n{exc}",
            ha="center",
            va="center",
            color=PDF_TEXT,
            fontsize=10,
        )
    ax.axis("off")
    pdf.savefig(
        page,
        dpi=PDF_EXPORT_DPI,
        bbox_inches="tight",
        facecolor=PDF_BG,
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(page)


def build_tactical_pdf(figs, info, events, xg_data, ts):
    """Build an English PDF with white background and clear readable fonts."""
    import glob

    matplotlib.rcParams.update(
        {
            "figure.facecolor": PDF_BG,
            "axes.facecolor": PDF_SURFACE,
            "text.color": PDF_TEXT,
            "savefig.facecolor": PDF_BG,
            "savefig.edgecolor": "none",
            "savefig.dpi": OUTPUT_IMAGE_DPI,
            "figure.dpi": 300,
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )
    hn, an = info["home_name"], info["away_name"]
    stats = _ensure_match_stats_defaults(_collect_match_stats(info, events, xg_data))
    safe_hn = re.sub(r"[^A-Za-z0-9_]+", "_", str(hn)).strip("_")
    safe_an = re.sub(r"[^A-Za-z0-9_]+", "_", str(an)).strip("_")
    pdf_path = (
        f"{SAVE_DIR}/match_analysis_report_EN_FULL_{safe_hn}_vs_{safe_an}_{ts}.pdf"
    )

    ordered_catalog = [
        m for m in _report_catalog_order(info) if 1 <= int(m.get("idx", 0)) <= len(figs)
    ]
    covered = {int(m.get("idx", 0)) for m in ordered_catalog}
    for idx in range(1, len(figs) + 1):
        if idx not in covered:
            ordered_catalog.append(
                {
                    "idx": idx,
                    "section": "additional",
                    "team": None,
                    "kind": "additional_visual",
                    "title": f"Additional Visual {idx}",
                }
            )

    board_paths = sorted(glob.glob(os.path.join(SAVE_DIR, f"board_*_{ts}.png")))
    total_pages = 2 + len(ordered_catalog) + len(board_paths)
    page_num = 1
    console.print(
        "\n[bold cyan]  Writing ENGLISH full-visual PDF report...[/bold cyan]"
    )
    console.print(f"[bold cyan]  Building PDF: {pdf_path}[/bold cyan]")
    console.print(
        f"[dim]  PDF coverage: {len(ordered_catalog)} individual visuals + {len(board_paths)} grouped boards[/dim]"
    )

    with PdfPages(pdf_path) as pdf:
        _render_cover_page(pdf, info, stats, events, total_pages)
        page_num += 1
        _render_executive_summary_page(
            pdf, info, stats, events, xg_data, page_num, total_pages
        )
        page_num += 1
        for _i, meta in enumerate(ordered_catalog):
            idx = int(meta.get("idx", 0))
            next_meta = (
                ordered_catalog[_i + 1] if _i + 1 < len(ordered_catalog) else None
            )
            statline, commentary = _visual_tactical_note(
                meta, info, events, xg_data, stats, next_meta=next_meta
            )
            _render_visual_page(
                pdf,
                figs[idx - 1],
                info,
                meta,
                statline,
                commentary,
                page_num,
                total_pages,
                events=events,
            )
            page_num += 1
        for board_path in board_paths:
            title = (
                os.path.splitext(os.path.basename(board_path))[0]
                .replace("_", " ")
                .title()
            )
            _render_board_image_page(
                pdf, board_path, info, title, page_num, total_pages, events=events
            )
            page_num += 1
        d = pdf.infodict()
        _h_sc, _a_sc = _parse_scoreline(info, xg_data, events=events)
        d["Title"] = f"English Full Match Analysis Report: {hn} {_h_sc}-{_a_sc} {an}"
        d["Author"] = "Mostafa Saad"
        d["Subject"] = f"{info.get('competition', '')} - {info.get('date', '')}"
        d["Keywords"] = (
            "football match analysis, tactical report, WhoScored, xG, xT, English PDF"
        )
    console.print(
        f"\n[bold green]  English full-visual PDF saved -> {pdf_path}[/bold green]\n"
    )
    return pdf_path


if __name__ == "__main__":
    main()
