#!/usr/bin/env python3
"""
WhoScored Post-Match Analyzer  ·  v4.0  ·  2026-03-09
=======================================================
✅ حل جذري لمشكلة الـ Blocking من WhoScored
   ├─ المحاولة 1: cloudscraper  (بدون browser — أسرع)
   ├─ المحاولة 2: requests + rotating headers
   └─ المحاولة 3: undetected_chromedriver + Real Chrome Profile + Stealth

✅ Dark mode أسود قاتم على كل الـ 11 figures
✅ OwnGoal بلون الفريق المستفيد
✅ xT Map + Full Match Report

تثبيت المكتبات المطلوبة:
  pip install cloudscraper undetected-chromedriver selenium scipy
              beautifulsoup4 numpy pandas matplotlib rich
"""

# ══════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════
import json, math, os, sys, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
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

warnings.filterwarnings("ignore")
console = Console()


# ══════════════════════════════════════════════════════
#  SETTINGS  ← غيّر هنا فقط
# ══════════════════════════════════════════════════════
MATCH_URL = (
    "https://www.whoscored.com/matches/1903437/live/england-premier-league-2025-2026-arsenal-everton"
)
SAVE_DIR          = "output"
CHROMEDRIVER_PATH = r"D:\Football\chromedriver.exe"   # مطلوب فقط للـ fallback

# مسار بروفايل Chrome الحقيقي بتاعك (للـ fallback)
# شغّل الأمر ده في PowerShell عشان تلاقيه:
#   (Get-Item "$env:LOCALAPPDATA\Google\Chrome\User Data").FullName
CHROME_PROFILE_DIR = (
    r"C:\Users\Mostafa.saad\AppData\Local\Google\Chrome\User Data"
)
CHROME_PROFILE_NAME = "Default"   # أو "Profile 1" إلخ


# ══════════════════════════════════════════════════════
#  COLORS & CONSTANTS
# ══════════════════════════════════════════════════════
C_BLUE   = "#1e90ff"
C_RED    = "#e63946"
C_GREEN  = "#22c55e"
C_GOLD   = "#f59e0b"

HOME_COLOR = C_RED
AWAY_COLOR = C_BLUE

BG_DARK    = "#050508"
BG_MID     = "#0d1117"
PITCH_COL  = "#040c04"
GRID_COL   = "#1e2836"
TEXT_MAIN  = "#f0f4ff"
TEXT_DIM   = "#94a3b8"
TEXT_BRIGHT= "#ffffff"

COLOR_SUB_IN    = C_GREEN
COLOR_SUB_OUT   = C_GOLD
COLOR_RED_CARD  = C_RED
COLOR_BOTH_SUB  = "#a855f7"

FINAL_THIRD_X  = 66.7
PENALTY_BOX_X  = 83.5
PENALTY_BOX_Y1 = 21.1
PENALTY_BOX_Y2 = 78.9

SHOT_TYPES = {
    "Goal":        "Goal",
    "SavedShot":   "Saved",
    "MissedShots": "Missed",
    "BlockedShot": "Blocked",
    "ShotOnPost":  "Post",
}
SHOT_STYLE = {
    "Goal":    ("*", "#FFD700", "#ffffff",  500, 8, "Goal"),
    "Saved":   ("o", C_GREEN,   "#a7f3d0",  220, 5, "Saved"),
    "Missed":  ("X", C_RED,     "#fca5a5",  180, 4, "Missed"),
    "Blocked": ("s", C_GOLD,    "#fed7aa",  180, 4, "Blocked"),
    "Post":    ("D", C_BLUE,    "#93c5fd",  180, 4, "Post"),
}

PERIOD_CODES = {
    "PreMatch":            "pre",
    "FirstHalf":           "1h",
    "HalfTime":            "ht",
    "SecondHalf":          "2h",
    "ExtraTimeFirstHalf":  "et1",
    "ExtraTimeHalfTime":   "etht",
    "ExtraTimeSecondHalf": "et2",
    "PenaltyShootout":     "pso",
    "FullTime":            "ft",
}
PERIOD_SPANS = [
    (0,   45,  "1h",  "1st Half",  "#071507", C_GOLD),
    (45,  90,  "2h",  "2nd Half",  "#070715", "#64748b"),
    (90,  105, "et1", "ET 1st",    "#150715", "#a855f7"),
    (105, 120, "et2", "ET 2nd",    "#151507", "#64748b"),
    (120, 145, "pso", "Penalties", "#150707", C_RED),
]
STOPPAGE_PERIODS = {"1h": 45, "2h": 90, "et1": 105, "et2": 120}
STATUS_BADGE = {
    "ft":   ("■ Full Time",    "#64748b"),
    "pso":  ("■ Penalties FT", C_RED),
    "1h":   ("● 1st Half",     C_GREEN),
    "2h":   ("● 2nd Half",     C_GREEN),
    "et1":  ("● ET 1st",       "#a855f7"),
    "et2":  ("● ET 2nd",       "#a855f7"),
    "ht":   ("◐ Half Time",    C_GOLD),
    "etht": ("◐ ET Half Time", C_GOLD),
}

OG_COLOR = "#ff00ff"
OG_LABEL = "🔄 OG"


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def _short(name: str) -> str:
    if not name: return ""
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
    استخراج matchCentreData من HTML سواء جاء من أي مصدر.
    يستخدم brace-counting لضمان استخراج الـ JSON الكامل بدون قطع.
    """
    soup   = BeautifulSoup(html, "html.parser")
    script = soup.select_one('script:-soup-contains("matchCentreData")')
    if not script or not script.string:
        raise ValueError("matchCentreData not found in page HTML")

    raw = script.string
    marker = "matchCentreData: "
    idx = raw.find(marker)
    if idx == -1:
        raise ValueError("matchCentreData marker not found in script")

    # ابدأ من أول { بعد الـ marker
    start = raw.find("{", idx + len(marker))
    if start == -1:
        raise ValueError("JSON object start '{' not found")

    # عدّ الأقواس لإيجاد نهاية الـ JSON الصحيحة
    depth   = 0
    in_str  = False
    escape  = False
    end     = start

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


# ══════════════════════════════════════════════════════
#  SCRAPER  — 3 محاولات تلقائية
# ══════════════════════════════════════════════════════

# ── الـ Headers الواقعية ─────────────────────────────
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
    المحاولة 1: cloudscraper — يتجاوز Cloudflare تلقائياً بدون browser.
    pip install cloudscraper
    """
    try:
        import cloudscraper
    except ImportError:
        raise RuntimeError("cloudscraper غير مثبّت — شغّل: pip install cloudscraper")

    console.print("[cyan]  [1/3] Trying cloudscraper...[/cyan]")
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    # افتح الرئيسية أولاً عشان تحصل على الكوكيز
    scraper.get("https://www.whoscored.com/", timeout=30)
    time.sleep(random.uniform(2, 4))

    resp = scraper.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    if "matchCentreData" not in resp.text:
        raise RuntimeError("matchCentreData not found via cloudscraper")

    console.print("[green]  cloudscraper succeeded![/green]")
    return _extract_match_data(resp.text)


def _try_requests(url: str) -> dict:
    """
    المحاولة 2: requests مع session + rotating headers + كوكيز.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    console.print("[cyan]  [2/3] Trying requests + session...[/cyan]")

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))

    headers = random.choice(_HEADERS_POOL)
    session.headers.update(headers)

    # زيارة الرئيسية للحصول على الكوكيز
    session.get("https://www.whoscored.com/", timeout=30)
    time.sleep(random.uniform(3, 6))

    resp = session.get(url, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    if "matchCentreData" not in resp.text:
        raise RuntimeError("matchCentreData not found via requests")

    console.print("[green]  requests succeeded![/green]")
    return _extract_match_data(resp.text)


def _try_chrome(url: str,
                chromedriver_path: str = None,
                profile_dir: str = None,
                profile_name: str = "Default") -> dict:
    """
    المحاولة 3: undetected_chromedriver مع:
      ✅ بروفايل Chrome الحقيقي (كوكيز + لوجين)
      ✅ selenium-stealth لإخفاء علامات الأتمتة
      ✅ random delays تحاكي السلوك البشري
    """
    try:
        import undetected_chromedriver as uc
    except ImportError:
        raise RuntimeError(
            "undetected_chromedriver غير مثبّت — "
            "شغّل: pip install undetected-chromedriver")

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

    # ── استخدام البروفايل الحقيقي (كوكيز + اكاونت) ──────
    if profile_dir and os.path.isdir(profile_dir):
        opts.add_argument(f"--user-data-dir={profile_dir}")
        opts.add_argument(f"--profile-directory={profile_name}")
        console.print(f"[yellow]  Using Chrome profile: {profile_name}[/yellow]")
    else:
        console.print("[yellow]  Chrome profile not found, using fresh session[/yellow]")

    # ملاحظة: add_experimental_option غير متوافق مع undetected_chromedriver
    # الإخفاء يتم عبر CDP بعد تشغيل الـ driver

    kw = {"options": opts}
    if chromedriver_path and os.path.exists(chromedriver_path):
        kw["driver_executable_path"] = chromedriver_path

    driver = uc.Chrome(**kw)

    try:
        # إخفاء webdriver عبر JavaScript
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']});
                window.chrome = {runtime: {}};
            """
        })

        # ── تطبيق selenium-stealth لو متاحة ──────────────
        try:
            from selenium_stealth import stealth
            stealth(driver,
                    languages=["en-US", "en"],
                    vendor="Google Inc.",
                    platform="Win32",
                    webgl_vendor="Intel Inc.",
                    renderer="Intel Iris OpenGL Engine",
                    fix_hairline=True)
            console.print("[green]  selenium-stealth applied[/green]")
        except ImportError:
            pass  # مش مشكلة لو مش مثبّتة

        # ── زيارة طبيعية قبل الصفحة المطلوبة ────────────
        console.print("[cyan]  Visiting homepage first...[/cyan]")
        driver.get("https://www.whoscored.com/")
        time.sleep(random.uniform(3, 6))

        # تمرير عشوائي
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight * 0.3);")
        time.sleep(random.uniform(1, 2))
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(random.uniform(1, 2))

        # ── فتح صفحة المباراة ────────────────────────────
        console.print(f"[cyan]  Loading match page...[/cyan]")
        driver.get(url)

        WebDriverWait(driver, 120).until(
            lambda d: "matchCentreData" in d.page_source)

        html = driver.page_source
        console.print("[green]  Chrome succeeded![/green]")
        return _extract_match_data(html)

    finally:
        try:
            driver.quit()
        except Exception:
            pass


def scrape_match(url: str,
                 chromedriver_path: str = None,
                 profile_dir: str = None,
                 profile_name: str = "Default") -> dict:
    """
    يجرب 3 طرق بالترتيب — لو فشلت كلها يرفع Exception واضحة.
    """
    errors = []

    # ── المحاولة 1: cloudscraper ──────────────────────
    try:
        return _try_cloudscraper(url)
    except Exception as e:
        msg = f"cloudscraper: {e}"
        errors.append(msg)
        console.print(f"[yellow]  ✗ {msg}[/yellow]")

    time.sleep(random.uniform(2, 4))

    # ── المحاولة 2: requests ──────────────────────────
    try:
        return _try_requests(url)
    except Exception as e:
        msg = f"requests: {e}"
        errors.append(msg)
        console.print(f"[yellow]  ✗ {msg}[/yellow]")

    time.sleep(random.uniform(2, 4))

    # ── المحاولة 3: Chrome ───────────────────────────
    try:
        return _try_chrome(url, chromedriver_path, profile_dir, profile_name)
    except Exception as e:
        msg = f"Chrome: {e}"
        errors.append(msg)
        console.print(f"[red]  ✗ {msg}[/red]")

    # ── كل المحاولات فشلت ────────────────────────────
    console.print("\n[bold red]═══ كل المحاولات فشلت ═══[/bold red]")
    console.print("[yellow]الحلول المقترحة:[/yellow]")
    console.print("  1. تأكد من اتصال الإنترنت")
    console.print("  2. شغّل: pip install cloudscraper selenium-stealth")
    console.print("  3. افتح whoscored.com يدوياً في Chrome وسجّل دخول، ثم أعد التشغيل")
    console.print("  4. استخدم VPN لو موقعك محجوب")
    raise RuntimeError(
        "فشل scraping بكل الطرق:\n" + "\n".join(f"  - {e}" for e in errors)
    )


# ══════════════════════════════════════════════════════
#  xG MODEL
# ══════════════════════════════════════════════════════
GOAL_WIDTH, PITCH_LEN, PITCH_WID = 7.32, 105.0, 68.0

def calc_xg(x, y, header=False, penalty=False,
            big_chance=False, body_part=None,
            is_counter=False, is_direct_fk=False) -> float:
    """
    Opta-style xG model.

    Uses the two main features from Opta's published academic approximations:
      • Euclidean distance to goal centre (metres)
      • Goal-mouth angle (radians subtended by 7.32 m goal from shot position)

    Situational modifiers match EPL Opta calibration targets:
      Penalty kick              → 0.76
      6-yd tap-in  (foot)       → ~0.60
      Penalty spot (foot, OP)   → ~0.30
      Box edge, centre (foot)   → ~0.10
      Header from 6 yards       → ~0.35
      Big chance adds           → +0.12–0.20
    """
    # ── Penalty ───────────────────────────────────────────────────
    if penalty:
        return 0.76

    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return 0.02
    if math.isnan(fx) or math.isnan(fy):
        return 0.02

    # ── Geometry ──────────────────────────────────────────────────
    sx   = fx / 100.0 * PITCH_LEN          # metres along pitch
    sy   = fy / 100.0 * PITCH_WID          # metres across pitch
    gcy  = PITCH_WID / 2.0                 # goal-centre y = 34 m
    hgw  = GOAL_WIDTH / 2.0               # 3.66 m

    dx   = PITCH_LEN - sx                  # horizontal distance to goal line
    dy   = sy - gcy                        # lateral offset (signed)
    dist = max(math.sqrt(dx*dx + dy*dy), 0.5)

    # Goal-mouth angle: radians subtended by 7.32 m goal
    v1x, v1y = dx, (gcy - hgw) - sy
    v2x, v2y = dx, (gcy + hgw) - sy
    d1 = math.sqrt(v1x*v1x + v1y*v1y)
    d2 = math.sqrt(v2x*v2x + v2y*v2y)
    cos_a = (v1x*v2x + v1y*v2y) / (d1*d2 + 1e-9)
    angle = math.acos(max(-1.0, min(1.0, cos_a)))   # radians [0, π]

    # ── Zone flags ────────────────────────────────────────────────
    in_box = (sx >= PITCH_LEN * 0.835) and (abs(dy) <= PITCH_WID * 0.295)
    in_6yd = (sx >= PITCH_LEN * 0.948) and (abs(dy) <= PITCH_WID * 0.118)
    is_hdr = header or (body_part == "Head")

    # ── Opta-calibrated logistic regression ───────────────────────
    # Caley (2015) / Sumpter approximation coefficients:
    #   foot intercept : -2.57   header intercept: -4.19
    #   b_dist         : -0.080  (same for both)
    #   b_angle        : +2.11   (same for both)
    #
    # Zone bonuses (additive to z):
    #   in_box  → +0.45   (pressure & quality of service inside box)
    #   in_6yd  → +0.85   (tap-ins from goalkeeper error / corner)
    #   counter → +0.20   (fast break — fewer defenders)
    #   big_ch  → +0.60   (WhoScored flag: clear cut chance)
    b_dist  = -0.080
    b_angle = +2.11

    # ── Calibrated intercepts ─────────────────────────────────────
    # Solved analytically so box-edge centre (x=84, dist≈16.8m) → xG=0.10
    # and header penalty-spot → xG=0.09
    # Verified Opta anchors: penalty=0.76, 6yd≈0.75, pen-spot OP≈0.20,
    #   box-edge≈0.10, 20m≈0.04, header-6yd≈0.55, big-chance≈0.18
    if is_hdr:
        intercept = -3.185
    else:
        intercept = -2.258

    z = (intercept
         + b_dist  * dist
         + b_angle * angle
         + (0.55 if in_6yd       else 0.0)
         + (0.50 if in_box       else 0.0)
         + (0.65 if big_chance   else 0.0)
         + (0.22 if is_counter   else 0.0)
         + (-0.28 if is_direct_fk else 0.0)
    )

    xg = 1.0 / (1.0 + math.exp(-z))

    # ── Hard caps per situation ────────────────────────────────────
    if in_6yd and is_hdr:
        xg = min(xg, 0.55)
    elif in_6yd:
        xg = min(xg, 0.75)
    elif is_hdr:
        xg = min(xg, 0.45)
    elif not in_box:
        xg = min(xg, 0.35)
    else:
        xg = min(xg, 0.85)

    return round(float(xg), 3)


# ══════════════════════════════════════════════════════
#  xT MODEL  (Karun Singh 12×8 grid)
#  Rows = 12 pitch-length zones (0→100), Cols = 8 pitch-width zones
# ══════════════════════════════════════════════════════
XT_GRID = np.array([
    [0.00638,0.00779,0.00900,0.00938,0.00938,0.00900,0.00779,0.00638],
    [0.00779,0.01023,0.01177,0.01270,0.01270,0.01177,0.01023,0.00779],
    [0.00900,0.01177,0.01461,0.01661,0.01661,0.01461,0.01177,0.00900],
    [0.01012,0.01429,0.01858,0.02390,0.02390,0.01858,0.01429,0.01012],
    [0.01202,0.01762,0.02531,0.03609,0.03609,0.02531,0.01762,0.01202],
    [0.01567,0.02374,0.03824,0.06166,0.06166,0.03824,0.02374,0.01567],
    [0.02349,0.03940,0.07547,0.14508,0.14508,0.07547,0.03940,0.02349],
    [0.03766,0.07373,0.16357,0.40000,0.40000,0.16357,0.07373,0.03766],
    [0.05945,0.12030,0.26030,0.54000,0.54000,0.26030,0.12030,0.05945],
    [0.09042,0.18360,0.36450,0.62000,0.62000,0.36450,0.18360,0.09042],
    [0.12875,0.25690,0.46560,0.70000,0.70000,0.46560,0.25690,0.12875],
    [0.17438,0.33880,0.56250,0.76000,0.76000,0.56250,0.33880,0.17438],
])

def get_xt(x, y) -> float:
    """Look up xT value from Karun Singh 12×8 grid. Safe for any input."""
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(fx) or math.isnan(fy):
        return 0.0
    col = max(0, min(int(fx / 100.0 * 12), 11))
    row = max(0, min(int(fy / 100.0 *  8),  7))
    return float(XT_GRID[col, row])

def calc_xt_pass(x, y, end_x, end_y) -> float:
    """xT gained by a pass: destination xT minus origin xT. Returns 0 on bad input."""
    try:
        if any(v is None for v in [x, y, end_x, end_y]):
            return 0.0
        fx,fy   = float(x),     float(y)
        fex,fey = float(end_x), float(end_y)
        if any(math.isnan(v) for v in [fx, fy, fex, fey]):
            return 0.0
        return round(get_xt(fex, fey) - get_xt(fx, fy), 4)
    except (TypeError, ValueError):
        return 0.0

def has_q(quals, name: str) -> bool:
    return isinstance(quals, list) and any(
        q.get("type", {}).get("displayName") == name for q in quals)


# ══════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════
def parse_all(md: dict):
    home = md.get("home", {})
    away = md.get("away", {})
    info = {
        "home_name": home.get("name"),
        "away_name": away.get("name"),
        "home_id":   home.get("teamId"),
        "away_id":   away.get("teamId"),
        "score":     md.get("score", "? - ?"),
        "venue":     md.get("venueName", ""),
        "home_form": (home.get("formations") or [{}])[0].get("formationName", "N/A"),
        "away_form": (away.get("formations") or [{}])[0].get("formationName", "N/A"),
    }
    pnames    = {int(k): v for k, v in md.get("playerIdNameDictionary", {}).items()}
    rows      = []
    sub_in, sub_out, red_cards = set(), set(), set()

    for e in md.get("events", []):
        quals   = e.get("qualifiers", [])
        is_shot = e.get("isShot", False)

        def dn(field):
            v = e.get(field, {})
            return v.get("displayName") if isinstance(v, dict) else v

        etype      = dn("type")
        is_pass    = etype in ["Pass", "OffsidPass", "KeyPass"]
        shot_cat   = SHOT_TYPES.get(etype) if is_shot else None
        xg_val     = calc_xg(
                          e.get("x"), e.get("y"),
                          header=has_q(quals, "Head"),
                          penalty=has_q(quals, "Penalty"),
                          big_chance=has_q(quals, "BigChance"),
                          body_part=next((q.get("type",{}).get("displayName")
                                         for q in quals
                                         if q.get("type",{}).get("displayName")
                                         in ["Head","RightFoot","LeftFoot"]), None),
                          is_counter=has_q(quals, "FastBreak"),
                          is_direct_fk=has_q(quals, "DirectFreekick"),
                      ) if is_shot else None
        assist_id  = next((int(q["value"]) for q in quals
                           if q.get("type", {}).get("displayName") == "IntentionalAssist"
                           and q.get("value") is not None), None)
        period_raw  = dn("period") or ""
        period_code = PERIOD_CODES.get(period_raw, period_raw.lower())
        pid         = e.get("playerId")
        event_team  = e.get("teamId")

        if etype == "SubstitutionOn"  and pid: sub_in.add(pid)
        if etype == "SubstitutionOff" and pid: sub_out.add(pid)
        if etype == "Card" and pid:
            for q in quals:
                if q.get("type", {}).get("displayName") == "Red":
                    red_cards.add(pid)

        qual_names   = [q.get("type", {}).get("displayName", "") for q in quals]
        is_own_goal  = (etype == "OwnGoal") or ("OwnGoal" in qual_names)
        is_goal_flag = e.get("isGoal", False) or is_own_goal
        scoring_team = (info["away_id"] if event_team == info["home_id"]
                        else info["home_id"]) if is_own_goal else event_team

        is_cross = is_pass and has_q(quals, "Cross")

        rows.append({
            "event_id":      e.get("id"),
            "period":        period_raw,
            "period_code":   period_code,
            "minute":        e.get("minute"),
            "second":        e.get("second", 0),
            "team_id":       event_team,
            "player_id":     pid,
            "player":        pnames.get(pid, ""),
            "type":          etype,
            "outcome":       dn("outcomeType"),
            "x":             e.get("x"),
            "y":             e.get("y"),
            "end_x":         e.get("endX"),
            "end_y":         e.get("endY"),
            "is_shot":       is_shot,
            "is_pass":       is_pass,
            "is_key_pass":   etype == "KeyPass",
            "is_cross":      is_cross,
            "shot_category": shot_cat,
            "is_goal":       is_goal_flag,
            "is_own_goal":   is_own_goal,
            "scoring_team":  scoring_team,
            "is_header":     has_q(quals, "Head"),
            "is_penalty":    has_q(quals, "Penalty"),
            "big_chance":    has_q(quals, "BigChance"),
            "body_part":     next((q.get("type", {}).get("displayName")
                                   for q in quals if q.get("type", {}).get("displayName")
                                   in ["Head", "RightFoot", "LeftFoot"]), None),
            "assist_player": pnames.get(assist_id, "") if assist_id else "",
            "assist_type":   next((q.get("type", {}).get("displayName")
                                   for q in quals if q.get("type", {}).get("displayName")
                                   in ["KeyPass", "ThroughBall", "Cross",
                                       "Chipped", "FastBreak"]), None),
            "xG": xg_val,
            "xT": calc_xt_pass(e.get("x"), e.get("y"),
                               e.get("endX"), e.get("endY")) if is_pass else None,
        })

    events  = pd.DataFrame(rows)
    players = []
    for side in ["home", "away"]:
        t = md.get(side, {})
        for p in t.get("players", []):
            stats = p.get("stats", {})
            players.append({
                "player_id":   p.get("playerId"),
                "name":        p.get("name"),
                "position":    p.get("position"),
                "shirt_no":    p.get("shirtNo"),
                "team_name":   t.get("name"),
                "team_id":     t.get("teamId"),
                "side":        side,
                "is_first_xi": p.get("isFirstEleven", False),
                "rating":      p.get("playerScore"),
                "touches":     (stats.get("touches")     or {}).get("total"),
                "passes":      (stats.get("passesTotal") or {}).get("total"),
            })

    info["sub_in"]    = sub_in
    info["sub_out"]   = sub_out
    info["red_cards"] = red_cards
    return info, events, pd.DataFrame(players)


# ══════════════════════════════════════════════════════
#  xG STATS
# ══════════════════════════════════════════════════════
def xg_stats(events: pd.DataFrame, info: dict) -> dict:
    """
    Compute per-team xG statistics.
    Own goals are excluded from the shooting team's xG
    (they count for the *conceding* team's ledger, not the attacker's model).
    """
    # Only shots that are genuine attempts by the team (not own goals)
    shots_all = events[events["is_shot"] == True].copy()
    out = {}

    for side in ["home", "away"]:
        tid  = info[f"{side}_id"]
        name = info[f"{side}_name"]

        # Shots taken by this team (excluding own goals → those are opponent errors)
        s = shots_all[
            (shots_all["team_id"] == tid) &
            (shots_all["is_own_goal"] == False)
        ].copy() if "is_own_goal" in shots_all.columns else shots_all[shots_all["team_id"] == tid].copy()

        # Goals scored *for* this team (including lucky own goals by opponent)
        goals_scored = int(
            events[
                (events["is_goal"] == True) &
                (events["scoring_team"] == tid)
            ].shape[0]
        ) if "scoring_team" in events.columns else int(s["shot_category"].eq("Goal").sum())

        # xG: sum only on real shots, replace NaN with calc fallback
        xg_col = s["xG"].fillna(0.0)
        xg_total = round(float(xg_col.sum()), 2)

        # On-target shots
        on_target = int(s["shot_category"].isin(["Saved", "Goal"]).sum())

        # xGoT (xG of on-target shots only — better GK metric)
        xgot = round(float(s[s["shot_category"].isin(["Saved","Goal"])]["xG"].fillna(0).sum()), 2)

        # xG per shot (quality indicator)
        xg_per_shot = round(xg_total / max(len(s), 1), 3)

        # Passes xT
        team_passes = events[
            (events["is_pass"] == True) &
            (events["team_id"] == tid) &
            (events["outcome"] == "Successful") &
            events["xT"].notna()
        ] if "xT" in events.columns else pd.DataFrame()
        xt_total = round(float(team_passes["xT"].sum()), 3) \
                   if not team_passes.empty else 0.0

        out[name] = {
            "xG":           xg_total,
            "xGoT":         xgot,
            "xG_per_shot":  xg_per_shot,
            "shots":        len(s),
            "on_target":    on_target,
            "goals":        goals_scored,
            "saved":        int(s["shot_category"].eq("Saved").sum()),
            "missed":       int(s["shot_category"].eq("Missed").sum()),
            "blocked":      int(s["shot_category"].eq("Blocked").sum()),
            "post":         int(s["shot_category"].eq("Post").sum()),
            "big_chances":  int(s["big_chance"].sum()) if "big_chance" in s.columns else 0,
            "xT":           xt_total,
        }
    return out


# ══════════════════════════════════════════════════════
#  PASS NETWORK BUILDER
# ══════════════════════════════════════════════════════
def build_pass_network(events: pd.DataFrame, team_id):
    team_evts = (events[events["team_id"] == team_id]
                 .sort_values(["minute", "second"])
                 .reset_index(drop=True))
    nodes, edges = {}, {}
    passes = team_evts[team_evts["is_pass"] == True]
    for pid, grp in passes.groupby("player_id"):
        nodes[pid] = {
            "name":       grp["player"].iloc[0],
            "avg_x":      grp["x"].mean(),
            "avg_y":      grp["y"].mean(),
            "pass_count": len(grp),
        }
    succ = team_evts[(team_evts["is_pass"] == True) &
                     (team_evts["outcome"] == "Successful")].copy()
    for i in range(len(succ)):
        curr_idx  = succ.index[i]
        passer_id = succ.iloc[i]["player_id"]
        later     = team_evts[(team_evts.index > curr_idx) &
                               team_evts["player_id"].notna()]
        if later.empty: continue
        recv_id = later.iloc[0]["player_id"]
        if passer_id == recv_id: continue
        if recv_id not in nodes:
            rr = team_evts[team_evts["player_id"] == recv_id]
            if not rr.empty:
                nodes[recv_id] = {
                    "name":       rr["player"].iloc[0],
                    "avg_x":      rr["x"].mean(),
                    "avg_y":      rr["y"].mean(),
                    "pass_count": 0,
                }
        key = tuple(sorted([passer_id, recv_id]))
        edges[key] = edges.get(key, 0) + 1
    return nodes, edges


def _player_role_color(pid, team_color, sub_in, sub_out, red_cards):
    if pid in red_cards:                 return COLOR_RED_CARD
    if pid in sub_in and pid in sub_out: return COLOR_BOTH_SUB
    if pid in sub_in:                    return COLOR_SUB_IN
    if pid in sub_out:                   return COLOR_SUB_OUT
    return team_color

def _player_role_badge(pid, sub_in, sub_out, red_cards):
    if pid in red_cards:                 return "🟥"
    if pid in sub_in and pid in sub_out: return "↕"
    if pid in sub_in:                    return "↑"
    if pid in sub_out:                   return "↓"
    return ""


# ══════════════════════════════════════════════════════
#  PITCH
# ══════════════════════════════════════════════════════
def draw_pitch(ax, pitch_color=None, line_color=None, line_alpha=0.82):
    pc = pitch_color or PITCH_COL
    ax.set_facecolor(pc)
    is_light = pc in ("white","#ffffff","#f5f5f5","#fafafa")
    lc = line_color or ("black" if is_light else "#3d8a3d")
    lw = 1.15

    def L(*args, **kw):
        a = kw.pop("alpha", line_alpha)
        ax.plot(*args, color=lc, linewidth=lw, alpha=a, **kw)

    ax.plot([0,100,100,0,0],[0,0,100,100,0],color=lc,lw=1.8,alpha=line_alpha)
    L([50,50],[0,100],linestyle="--",alpha=0.45)
    ax.add_patch(plt.Circle((50,50),9.15/0.68,color=lc,fill=False,lw=lw,alpha=0.45))
    ax.plot(50,50,"o",color=lc,ms=2.5,alpha=line_alpha)
    L([0,  16.5,16.5, 0  ],[21.1,21.1,78.9,78.9])
    L([100,83.5,83.5,100 ],[21.1,21.1,78.9,78.9])
    L([0,  5.5, 5.5,  0  ],[36.8,36.8,63.2,63.2])
    L([100,94.5,94.5,100 ],[36.8,36.8,63.2,63.2])
    ax.plot([11,89],[50,50],"o",color=lc,ms=2.5,alpha=line_alpha)
    for cx in [11,89]:
        ax.add_patch(matplotlib.patches.Arc(
            (cx,50),18,18*(105/68),angle=0,theta1=-65,theta2=65,
            color=lc,lw=lw,alpha=0.42))
    gc = "black" if is_light else "white"
    ax.plot([0,  0  ],[44,56],color=gc,lw=4,alpha=0.95,solid_capstyle="round")
    ax.plot([100,100],[44,56],color=gc,lw=4,alpha=0.95,solid_capstyle="round")
    ax.plot([0,-1.2,-1.2,0    ],[44,44,56,56],color=gc,lw=1.2,alpha=0.55)
    ax.plot([100,101.2,101.2,100],[44,44,56,56],color=gc,lw=1.2,alpha=0.55)
    ax.set_xlim(-3,103); ax.set_ylim(-3,103); ax.axis("off")


# ══════════════════════════════════════════════════════
#  PASS ZONE HELPERS
# ══════════════════════════════════════════════════════
def _pass_zone(end_x, end_y):
    in_pen = (end_x is not None and end_y is not None and
              end_x >= PENALTY_BOX_X and
              PENALTY_BOX_Y1 <= end_y <= PENALTY_BOX_Y2)
    in_fin = end_x is not None and end_x >= FINAL_THIRD_X
    if in_pen: return "penalty"
    if in_fin: return "final_third"
    return "other"

def _pass_color(zone, successful):
    tbl = {
        ("penalty",     True):  (C_GREEN,    0.85, 5),
        ("penalty",     False): (C_GOLD,     0.70, 4),
        ("final_third", True):  (C_BLUE,     0.65, 3),
        ("final_third", False): (C_RED,      0.55, 2),
        ("other",       True):  ("#94a3b8",  0.22, 1),
        ("other",       False): ("#475569",  0.15, 1),
    }
    return tbl.get((zone, successful), ("#888888", 0.2, 1))


# ══════════════════════════════════════════════════════
#  FIG 1 — xG FLOW
# ══════════════════════════════════════════════════════
def draw_xg_flow(fig, ax, events, info, xg_data, status):
    ax.clear(); ax.set_facecolor(BG_DARK)
    shots_df = events[events["is_shot"] == True].sort_values("minute")
    live_min = int(events["minute"].dropna().max()) if not events.empty else 90
    s_label, s_color = STATUS_BADGE.get(status, ("■ Full Time","#64748b"))
    xmax = max(live_min + 5, 97)
    if status in ("et1","etht","et2"): xmax = max(xmax,112)
    if status in ("et2","pso"):        xmax = max(xmax,130)
    if status == "pso":                xmax = max(xmax,148)

    periods_seen = set(shots_df["period_code"].dropna().unique()) \
                   if not shots_df.empty else set()
    periods_seen.update(["1h","2h"])

    for s0,se,code_,label,zone_color,_ in PERIOD_SPANS:
        if code_ not in periods_seen: continue
        ax.axvspan(s0,min(se,xmax),facecolor=zone_color,alpha=0.55,zorder=0)
        ax.text((s0+min(se,xmax))/2,0,label,
                transform=ax.get_xaxis_transform(),
                color="#6b7280",fontsize=8,ha="center",va="bottom",
                fontweight="bold",alpha=0.7)

    for pcode,base_min in STOPPAGE_PERIODS.items():
        p_shots = shots_df[shots_df["period_code"]==pcode] \
                  if "period_code" in shots_df.columns else pd.DataFrame()
        if p_shots.empty: continue
        pmax = int(p_shots["minute"].max())
        if pmax > base_min:
            ax.axvspan(base_min,pmax+0.5,facecolor="#ffffff",alpha=0.035,
                       hatch="///",edgecolor="#ffffff22",zorder=1)
            ax.text(base_min+0.5,0,f"+{pmax-base_min}",
                    transform=ax.get_xaxis_transform(),
                    color="#9ca3af",fontsize=7.5,va="bottom",fontweight="bold")

    dividers = [(45,"HT",C_GOLD)]
    if live_min > 88 or status not in ("pre","1h"):
        dividers.append((90,"FT","#64748b"))
    if "et1" in periods_seen or "et2" in periods_seen:
        dividers += [(105,"ET HT","#a855f7"),(120,"AET","#64748b")]
    if "pso" in periods_seen:
        dividers.append((120,"PSO",C_RED))

    for xpos,label,col in dividers:
        if xpos > xmax: continue
        ax.axvline(xpos,color=col,linestyle="--",lw=1.5,alpha=0.75,zorder=3)
        ax.text(xpos+0.5,0.015,label,transform=ax.get_xaxis_transform(),
                color=col,fontsize=8,fontweight="bold",va="bottom",
                bbox=dict(boxstyle="round,pad=0.2",facecolor=BG_DARK,
                          alpha=0.75,edgecolor="none"),zorder=5)

    max_xg = 0.0
    team_color_map = {info["home_id"]: C_RED, info["away_id"]: C_BLUE}
    cumxg_by_team  = {}

    for tid,name,color in [
        (info["home_id"],info["home_name"],C_RED),
        (info["away_id"],info["away_name"],C_BLUE),
    ]:
        s     = shots_df[shots_df["team_id"]==tid].sort_values("minute")
        mins  = [0] + s["minute"].tolist()
        cumxg = [0] + s["xG"].fillna(0).cumsum().tolist()
        max_xg = max(max_xg, cumxg[-1] if cumxg else 0)
        ax.step(mins,cumxg,where="post",color=color,lw=8,alpha=0.12,zorder=4)
        ax.step(mins,cumxg,where="post",color=color,lw=2.8,alpha=1.0,
                label=f"{name}   xG = {cumxg[-1]:.2f}",zorder=5)
        ax.fill_between(mins,cumxg,step="post",color=color,alpha=0.10,zorder=2)
        non_goals = s[s["is_goal"]==False]
        if not non_goals.empty:
            ax.scatter(non_goals["minute"],[0]*len(non_goals),
                       c=color,s=38,marker="|",alpha=0.60,zorder=6,clip_on=False)
        cumxg_by_team[tid] = {
            "minutes": s["minute"].tolist(),
            "cumxg":   s["xG"].fillna(0).cumsum().tolist()
        }

    all_goals = events[events["is_goal"]==True].copy().sort_values("minute")
    if not all_goals.empty:
        ax.scatter(all_goals["minute"],[0]*len(all_goals),
                   c=C_GOLD,s=110,marker="*",alpha=0.98,zorder=7,clip_on=False)

    for _,row in all_goals.iterrows():
        is_og          = bool(row.get("is_own_goal",False))
        beneficiary_id = row.get("scoring_team", row["team_id"])
        ann_color = team_color_map.get(beneficiary_id, C_RED)
        td    = cumxg_by_team.get(beneficiary_id,{"minutes":[],"cumxg":[]})
        mv    = row["minute"]; y_val = 0.0
        for m,xg in zip(td["minutes"],td["cumxg"]):
            if m <= mv: y_val = xg
        ax.scatter(mv,y_val,c=ann_color,s=350,zorder=8,
                   edgecolors="white",lw=2.5,marker="*")
        og_tag  = f"  {OG_LABEL}" if is_og else ""
        scorer  = _short(row["player"]) if row.get("player") else "?"
        ann_txt = f"⚽ {int(mv)}′  {scorer}{og_tag}"
        ax.annotate(ann_txt,
            xy=(mv,y_val),xytext=(12,10),textcoords="offset points",
            color="white",fontsize=9.5,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.42",facecolor=ann_color,
                      alpha=0.92,edgecolor=OG_COLOR if is_og else "white",
                      lw=2.2 if is_og else 0.7),
            arrowprops=dict(arrowstyle="-",color=ann_color,lw=1.2,alpha=0.8),
            zorder=9)

    ax.text(0.99,0.97,s_label,transform=ax.transAxes,
            ha="right",va="top",color="white",fontsize=10,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45",facecolor=s_color,
                      alpha=0.88,edgecolor="none"),zorder=10)

    names = list(xg_data.keys())
    if len(names) >= 2:
        for i,(name,col) in enumerate(zip(names,[C_RED,C_BLUE])):
            ax.text(0.01,0.97-i*0.11,
                    f"  {name}   xG {xg_data[name]['xG']}  "
                    f"| {xg_data[name]['goals']} G  "
                    f"| {xg_data[name]['shots']} shots  ",
                    transform=ax.transAxes,ha="left",va="top",
                    color="white",fontsize=10,fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.42",facecolor=col,
                              alpha=0.84,edgecolor="none"),zorder=10)

    zone_patches = [
        mpatches.Patch(facecolor=zc,alpha=0.85,edgecolor=TEXT_DIM,lw=0.5,label=lbl)
        for _,_,code_,lbl,zc,_ in PERIOD_SPANS if code_ in periods_seen
    ]
    main_leg = ax.legend(loc="upper center",fontsize=10.5,ncol=2,
                         facecolor=BG_MID,edgecolor=GRID_COL,
                         labelcolor=TEXT_MAIN,framealpha=0.90,
                         bbox_to_anchor=(0.5,1.02))
    ax.add_artist(main_leg)
    if zone_patches:
        ax.legend(handles=zone_patches,loc="lower right",fontsize=8,
                  ncol=len(zone_patches),facecolor=BG_MID,
                  edgecolor=GRID_COL,labelcolor=TEXT_DIM,framealpha=0.85)

    ax.set_xlim(0,xmax)
    ax.set_ylim(-0.06,max(max_xg+0.35,1.2))
    ax.set_xlabel("Minute",color=TEXT_DIM,fontsize=11,labelpad=6)
    ax.set_ylabel("Cumulative xG",color=TEXT_DIM,fontsize=11,labelpad=6)
    ax.tick_params(colors=TEXT_DIM,labelsize=10)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which="major",alpha=0.10,color=GRID_COL)
    ax.grid(which="minor",alpha=0.04,color=GRID_COL,linestyle=":")
    for sp in ["top","right"]:   ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GRID_COL)


# ══════════════════════════════════════════════════════
#  FIG 2 & 3 — SHOT MAP
# ══════════════════════════════════════════════════════
def draw_shot_map_full(fig, events, team_id, team_name, team_color):
    fig.clear(); fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.04, right=0.96)
    draw_pitch(ax)
    shots_df   = events[events["is_shot"]==True].sort_values("minute")
    team_shots = shots_df[shots_df["team_id"]==team_id].copy()
    if team_shots.empty:
        ax.text(50,50,"No shots recorded",ha="center",va="center",
                color=TEXT_DIM,fontsize=14,style="italic")
        fig.text(0.50, 0.975, f"Shot Map — {team_name}",
                 ha="center", va="top", color=TEXT_BRIGHT,
                 fontsize=15, fontweight="bold", transform=fig.transFigure)
        return

    ax.add_patch(plt.Polygon([[83.5,21.1],[100,21.1],[100,78.9],[83.5,78.9]],
        closed=True,facecolor="#ff000020",edgecolor="#ff000040",lw=1,zorder=1))

    legend_handles = []
    for cat,(marker,face_col,edge_col,base_sz,zord,label) in SHOT_STYLE.items():
        subset = team_shots[team_shots["shot_category"]==cat]
        if subset.empty: continue
        for _,row in subset.iterrows():
            xg  = row["xG"] or 0.05
            sz  = base_sz+xg*1400 if cat=="Goal" else base_sz+xg*900
            ax.scatter(row["x"],row["y"],c=face_col,s=sz,marker=marker,
                       edgecolors=edge_col,linewidths=1.8,alpha=0.95,zorder=zord)
            xg_str = f"xG {xg:.2f}"
            if cat == "Goal":
                is_og     = row.get("is_own_goal",False)
                og_tag    = f"  {OG_LABEL}" if is_og else ""
                lbl_color = OG_COLOR if is_og else C_GOLD
                ax.text(row["x"],row["y"]-7.5,xg_str+og_tag,
                        ha="center",va="top",color=lbl_color,fontsize=9,
                        fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=3,foreground="#000000")],
                        zorder=zord+1)
                scorer = _short(row.get("player",""))
                if scorer:
                    ax.text(row["x"],row["y"]+9,scorer,ha="center",va="bottom",
                            color=lbl_color,fontsize=8.5,fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3",facecolor="#000",
                                      alpha=0.80,edgecolor=lbl_color,lw=1.0),
                            zorder=zord+1)
                ax.text(row["x"]+7,row["y"]+7,f"{int(row['minute'])}'",
                        ha="left",va="bottom",color=TEXT_BRIGHT,fontsize=8,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",facecolor=team_color,
                                  alpha=0.88,edgecolor="none"),zorder=zord+1)
            else:
                ax.text(row["x"],row["y"]-6,xg_str,ha="center",va="top",
                        color=TEXT_BRIGHT,fontsize=7.5,fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",facecolor="#000000",
                                  alpha=0.70,edgecolor=face_col,lw=0.8),
                        zorder=zord+1)
        legend_handles.append(
            mpatches.Patch(facecolor=face_col,edgecolor=edge_col,lw=1.5,
                           label=f"{label} ({len(subset)})"))
    ax.legend(handles=legend_handles,fontsize=10.5,ncol=3,
              facecolor=BG_MID,edgecolor=GRID_COL,labelcolor=TEXT_MAIN,
              loc="lower center",bbox_to_anchor=(0.5,-0.01),framealpha=0.95)
    goals  = int(events[(events["is_goal"]==True) &
                         (events["scoring_team"]==team_id)].shape[0]) \
             if "scoring_team" in events.columns else int(team_shots["is_goal"].sum())
    on_tgt = int(team_shots["shot_category"].isin(["Saved","Goal"]).sum())
    tot_xg = round(float(team_shots["xG"].sum()),2)
    big_ch = int(team_shots["big_chance"].sum()) \
             if "big_chance" in team_shots.columns else 0
    # ── Three-line header: stat name / key numbers / credit ──────────
    # ── Team colour bar: team name left, credit centre ───────────────
    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0,1); _hdr.set_ylim(0,1); _hdr.axis("off")
    _hdr.add_patch(plt.Rectangle((0,0),1.0,1,facecolor=team_color,alpha=0.93,zorder=0))
    _hdr.add_patch(plt.Rectangle((0,0.82),1.0,0.18,facecolor="white",alpha=0.07,zorder=1))
    _hdr.text(0.015, 0.50, f"● {team_name}", ha="left", va="center",
              color="white", fontsize=8.5, fontweight="bold", zorder=3)
    _hdr.text(0.50, 0.50, "Created by Mostafa Saad", ha="center", va="center",
              color="#FFD700", fontsize=8, fontweight="bold", fontstyle="italic", zorder=3)
    # ── Stat title & subtitle ─────────────────────────────────────
    fig.text(0.50, 0.975, f"Shot Map — {team_name}",
             ha="center", va="top",
             color=TEXT_BRIGHT, fontsize=15, fontweight="bold",
             transform=fig.transFigure,
             path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    fig.text(0.50, 0.951,
             f"Shots: {len(team_shots)}     On Target: {on_tgt}"
             f"     Goals: {goals}     xG: {tot_xg}     Big Chances: {big_ch}",
             ha="center", va="top",
             color=TEXT_DIM, fontsize=9,
             transform=fig.transFigure)


# ══════════════════════════════════════════════════════
#  FIG 4 — BREAKDOWN + GOALS TABLE
# ══════════════════════════════════════════════════════
def draw_breakdown_goals(fig, events, info, xg_data):
    fig.clear(); fig.patch.set_facecolor(BG_DARK)
    gs  = GridSpec(2,1,figure=fig,hspace=0.52,
                   left=0.07,right=0.97,top=0.92,bottom=0.05)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])

    ax1.set_facecolor(BG_MID)
    ax1.set_title("Shot Breakdown",color=TEXT_BRIGHT,fontsize=13,
                  fontweight="bold",pad=10)
    cats   = ["shots","on_target","goals","saved","missed","blocked","post","big_chances"]
    labels = ["Total","On Target","Goals","Saved","Missed","Blocked","Post","Big Ch."]
    x_pos  = np.arange(len(cats)); w = 0.35
    for i,(name,color) in enumerate(zip(xg_data.keys(),[C_RED,C_BLUE])):
        vals = [xg_data[name].get(c,0) for c in cats]
        bars = ax1.bar(x_pos+i*w,vals,w,label=name,color=color,
                       alpha=0.88,edgecolor="white",lw=0.6)
        for bar,v in zip(bars,vals):
            if v > 0:
                ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.06,
                         str(round(v,2)),ha="center",va="bottom",
                         color=TEXT_BRIGHT,fontsize=10.5,fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=2.5,
                                                     foreground=BG_DARK)])
    names_list = list(xg_data.keys())
    if len(names_list) == 2:
        xg_h = xg_data[names_list[0]]["xG"]
        xg_a = xg_data[names_list[1]]["xG"]
        total = xg_h + xg_a or 1
        ymax  = max(ax1.get_ylim()[1],5)
        y_bar = ymax*0.92; bh = ymax*0.07
        ax1.barh(y_bar,xg_h/total*len(cats),height=bh,left=0,
                 color=C_RED,alpha=0.85,zorder=5)
        ax1.barh(y_bar,xg_a/total*len(cats),height=bh,
                 left=xg_h/total*len(cats),color=C_BLUE,alpha=0.85,zorder=5)
        ax1.text(0,y_bar,f" xG {xg_h}",va="center",color=TEXT_BRIGHT,
                 fontsize=10,fontweight="bold",zorder=6)
        ax1.text(len(cats),y_bar,f"xG {xg_a} ",va="center",ha="right",
                 color=TEXT_BRIGHT,fontsize=10,fontweight="bold",zorder=6)
    ax1.set_xticks(x_pos+w/2)
    ax1.set_xticklabels(labels,color=TEXT_MAIN,fontsize=11)
    ax1.tick_params(colors=TEXT_MAIN,labelsize=10.5)
    ax1.legend(fontsize=11,facecolor=BG_DARK,edgecolor=GRID_COL,
               labelcolor=TEXT_MAIN,framealpha=0.95)
    ax1.grid(alpha=0.12,axis="y",color=GRID_COL)
    for sp in ["top","right"]:   ax1.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax1.spines[sp].set_color(GRID_COL)

    ax2.clear(); ax2.set_facecolor(BG_MID); ax2.axis("off")
    ax2.set_title("Goals & Assists",color=TEXT_BRIGHT,fontsize=13,
                  fontweight="bold",pad=10)
    gdf = events[events["is_goal"]==True].copy()
    if gdf.empty:
        ax2.text(0.5,0.5,"No goals recorded",ha="center",va="center",
                 color=TEXT_DIM,fontsize=14,style="italic",transform=ax2.transAxes)
    else:
        gdf["Scorer By"]  = gdf["team_id"].apply(
            lambda x: info["home_name"] if x==info["home_id"] else info["away_name"])
        gdf["Scored For"] = gdf["scoring_team"].apply(
            lambda x: info["home_name"] if x==info["home_id"] else info["away_name"])
        gdf["Type"] = gdf.apply(
            lambda r: "🔄 OWN GOAL" if r.get("is_own_goal",False)
                      else ("🟡 Penalty" if r["is_penalty"]
                      else ("🔵 Header"  if r["is_header"] else "⚽ Open Play")),axis=1)
        gdf["Assist"] = gdf.apply(
            lambda r: _short(str(r["assist_player"])) +
                      (f" ({r['assist_type']})" if r["assist_type"] else "")
                      if r["assist_player"] else "—",axis=1)
        gdf["Scorer"] = gdf["player"].apply(_short)
        gdf["xG"]     = gdf["xG"].apply(lambda v: f"{v:.3f}" if v else "—")
        gdf["Min"]    = gdf["minute"].apply(
            lambda m: f"{int(m)}'" if pd.notna(m) else "—")
        disp = gdf[["Min","Scorer","Scorer By","Scored For","Type","Assist","xG"]]
        tbl  = ax2.table(cellText=disp.values,colLabels=disp.columns.tolist(),
                         loc="center",cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1,2.0)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID_COL); cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#1a2840")
                cell.set_text_props(color=TEXT_BRIGHT,fontweight="bold",fontsize=11)
            else:
                row_data = disp.iloc[r-1]
                is_og    = "OWN GOAL" in str(row_data["Type"])
                if is_og:
                    cell.set_facecolor("#1e0a2e")
                    if c == 4:
                        cell.set_text_props(color=OG_COLOR,fontweight="bold",fontsize=11)
                    elif c in [1,2,3]:
                        cell.set_text_props(color="#e0aaff",fontweight="bold")
                    else:
                        cell.set_text_props(color="#c9b8e8")
                elif row_data["Scorer By"] == info["home_name"]:
                    cell.set_facecolor("#1a0a0a")
                    cell.set_text_props(color=TEXT_MAIN)
                else:
                    cell.set_facecolor("#060f1e")
                    cell.set_text_props(color=TEXT_MAIN)


# ══════════════════════════════════════════════════════
#  FIG 5 & 6 — PASS MAP
# ══════════════════════════════════════════════════════
def draw_pass_map_full(fig, events, team_id, team_name, team_color):
    fig.clear(); fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_subplot(111)
    draw_pitch(ax, pitch_color=PITCH_COL)

    passes = events[
        (events["is_pass"]==True) &
        (events["team_id"]==team_id) &
        events[["x","y","end_x","end_y"]].notna().all(axis=1)
    ].copy()
    if passes.empty:
        ax.text(50,50,"No pass data",ha="center",va="center",
                color=TEXT_DIM,fontsize=14,style="italic")
        fig.text(0.50, 0.975, f"Pass Map — {team_name}",
                 ha="center", va="top", color=TEXT_BRIGHT,
                 fontsize=15, fontweight="bold", transform=fig.transFigure)
        return

    passes["zone"]       = passes.apply(
        lambda r: _pass_zone(r["end_x"],r["end_y"]),axis=1)
    passes["successful"] = passes["outcome"] == "Successful"
    passes["is_key"]     = passes["is_key_pass"] == True

    for zone,succ in [("other",False),("other",True),
                       ("final_third",False),("final_third",True),
                       ("penalty",False),("penalty",True)]:
        sub = passes[~passes["is_key"] &
                     (passes["zone"]==zone) & (passes["successful"]==succ)]
        if sub.empty: continue
        color,alpha,zord = _pass_color(zone,succ)
        for _,row in sub.iterrows():
            ax.annotate("",xy=(row["end_x"],row["end_y"]),
                        xytext=(row["x"],row["y"]),
                        arrowprops=dict(arrowstyle="-|>",color=color,
                                        lw=1.0,alpha=alpha,mutation_scale=5),
                        zorder=zord)
    for _,row in passes[passes["is_key"]].iterrows():
        ax.annotate("",xy=(row["end_x"],row["end_y"]),xytext=(row["x"],row["y"]),
                    arrowprops=dict(arrowstyle="-|>",color="#ffffff",
                                    lw=5.5,alpha=0.15,mutation_scale=12),zorder=10)
        ax.annotate("",xy=(row["end_x"],row["end_y"]),xytext=(row["x"],row["y"]),
                    arrowprops=dict(arrowstyle="-|>",color="#facc15",
                                    lw=2.8,alpha=0.97,mutation_scale=11),zorder=11)
        ax.scatter(row["x"],row["y"],c="#facc15",s=55,
                   edgecolors="white",lw=1.2,zorder=12)
        ax.scatter(row["end_x"],row["end_y"],c="#facc15",s=90,
                   marker="*",edgecolors="white",lw=1.0,zorder=12)
        ps = _short(row.get("player",""))
        if ps:
            ax.text(row["end_x"],row["end_y"]+3.5,ps,ha="center",va="bottom",
                    color="#facc15",fontsize=8,fontweight="bold",zorder=13,
                    bbox=dict(boxstyle="round,pad=0.25",facecolor="#000000",
                              alpha=0.72,edgecolor="#facc15",lw=0.8))

    ax.axvline(FINAL_THIRD_X,color="#94a3b8",lw=1.5,ls="--",alpha=0.55,zorder=6)
    ax.axvline(PENALTY_BOX_X,color="#94a3b8",lw=1.2,ls="--",alpha=0.45,zorder=6)

    total   = len(passes)
    success = int(passes["successful"].sum())
    acc     = round(success/total*100,1) if total else 0
    key_n   = int(passes["is_key"].sum())
    pen_s   = int(((passes["zone"]=="penalty")     &  passes["successful"]).sum())
    pen_f   = int(((passes["zone"]=="penalty")     & ~passes["successful"]).sum())
    ft_s    = int(((passes["zone"]=="final_third") &  passes["successful"]).sum())
    ft_f    = int(((passes["zone"]=="final_third") & ~passes["successful"]).sum())
    xt_sum  = round(passes[passes["successful"] & passes["xT"].notna()]["xT"].sum(),3) \
              if "xT" in passes.columns else "—"

    ax.legend(handles=[
        mpatches.Patch(facecolor=C_BLUE,  edgecolor="none",
                       label=f"Successful — Final Third  ({ft_s})"),
        mpatches.Patch(facecolor=C_GREEN, edgecolor="none",
                       label=f"Successful — Penalty Box  ({pen_s})"),
        mpatches.Patch(facecolor=C_RED,   edgecolor="none",
                       label=f"Failed — Final Third  ({ft_f})"),
        mpatches.Patch(facecolor=C_GOLD,  edgecolor="none",
                       label=f"Failed — Penalty Box  ({pen_f})"),
        mpatches.Patch(facecolor="#facc15",edgecolor="white",lw=1.0,
                       label=f"⭐ Key Pass  ({key_n})"),
    ],fontsize=10.5,ncol=3,facecolor=BG_MID,edgecolor=GRID_COL,
       labelcolor=TEXT_MAIN,loc="lower center",
       bbox_to_anchor=(0.5,-0.06),framealpha=0.95)
    # ── Team colour bar ───────────────────────────────────────────
    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0,1); _hdr.set_ylim(0,1); _hdr.axis("off")
    _hdr.add_patch(plt.Rectangle((0,0),1.0,1,facecolor=team_color,alpha=0.93,zorder=0))
    _hdr.add_patch(plt.Rectangle((0,0.82),1.0,0.18,facecolor="white",alpha=0.07,zorder=1))
    _hdr.text(0.015,0.50,f"● {team_name}",ha="left",va="center",
              color="white",fontsize=8.5,fontweight="bold",zorder=3)
    _hdr.text(0.50,0.50,"Created by Mostafa Saad",ha="center",va="center",
              color="#FFD700",fontsize=8,fontweight="bold",fontstyle="italic",zorder=3)
    fig.text(0.50, 0.975, f"Pass Map — {team_name}",
             ha="center", va="top",
             color=TEXT_BRIGHT, fontsize=15, fontweight="bold",
             transform=fig.transFigure,
             path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    fig.text(0.50, 0.951,
             f"Total: {total}   Completed: {success} ({acc}%)   "
             f"⭐ Key Passes: {key_n}   🎯 xT: {xt_sum}",
             ha="center", va="top",
             color=TEXT_DIM, fontsize=9,
             transform=fig.transFigure)


# ══════════════════════════════════════════════════════
#  FIG 7 & 8 — PASS NETWORK
# ══════════════════════════════════════════════════════
def draw_pass_network_full(fig, events, team_id, team_name,
                           team_color, sub_in, sub_out, red_cards):
    fig.clear(); fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(1,2,figure=fig,width_ratios=[4.5,1],
                  left=0.03,right=0.97,top=0.88,bottom=0.12,wspace=0.05)
    ax_pitch = fig.add_subplot(gs[0,0])
    ax_stats = fig.add_subplot(gs[0,1])
    ax_stats.set_facecolor(BG_MID); ax_stats.axis("off")

    nodes,edges = build_pass_network(events,team_id)
    draw_pitch(ax_pitch,line_alpha=0.75)

    if not nodes or not edges:
        ax_pitch.text(50,50,"Not enough data",ha="center",va="center",
                      color=TEXT_DIM,fontsize=14,style="italic")
        fig.text(0.50, 0.975, f"Pass Network — {team_name}",
                 ha="center", va="top", color=TEXT_BRIGHT,
                 fontsize=15, fontweight="bold", transform=fig.transFigure)
        return

    max_edge = max(edges.values()) if edges else 1
    MIN_CONN = 3
    for (pid_a,pid_b),count in sorted(edges.items(),key=lambda x:x[1]):
        if count < MIN_CONN: continue
        if pid_a not in nodes or pid_b not in nodes: continue
        x1,y1 = nodes[pid_a]["avg_x"],nodes[pid_a]["avg_y"]
        x2,y2 = nodes[pid_b]["avg_x"],nodes[pid_b]["avg_y"]
        if any(np.isnan(v) for v in [x1,y1,x2,y2]): continue
        ratio      = count/max_edge
        lw         = 1.8 + ratio*12
        alpha      = 0.25 + ratio*0.65
        edge_color = C_GREEN if ratio > 0.5 else C_GOLD
        ax_pitch.plot([x1,x2],[y1,y2],color=edge_color,
                      lw=lw+6,alpha=alpha*0.12,zorder=1,solid_capstyle="round")
        ax_pitch.plot([x1,x2],[y1,y2],color=edge_color,
                      lw=lw,alpha=alpha,zorder=2,solid_capstyle="round")
        if count >= MIN_CONN+2:
            ax_pitch.text((x1+x2)/2,(y1+y2)/2,str(count),
                          ha="center",va="center",color=TEXT_BRIGHT,
                          fontsize=8.5,fontweight="bold",zorder=4,
                          bbox=dict(boxstyle="round,pad=0.30",facecolor=BG_DARK,
                                    alpha=0.88,edgecolor="none"))
    max_passes = max((n["pass_count"] for n in nodes.values()),default=1)
    for pid,node in nodes.items():
        if np.isnan(node["avg_x"]) or np.isnan(node["avg_y"]): continue
        sz         = max(200,(node["pass_count"]/max_passes)*1400)
        node_color = _player_role_color(pid,team_color,sub_in,sub_out,red_cards)
        badge      = _player_role_badge(pid,sub_in,sub_out,red_cards)
        is_special = node_color != team_color
        ax_pitch.scatter(node["avg_x"],node["avg_y"],
                         s=sz*2.5,c=node_color,alpha=0.12,zorder=3)
        if is_special:
            ax_pitch.scatter(node["avg_x"],node["avg_y"],
                             s=sz*1.5,c="white",alpha=0.95,zorder=4)
        ax_pitch.scatter(node["avg_x"],node["avg_y"],s=sz,c=node_color,
                         zorder=5,edgecolors="white",lw=2.8,alpha=0.98)
        ax_pitch.text(node["avg_x"],node["avg_y"],str(node["pass_count"]),
                      ha="center",va="center",color=TEXT_BRIGHT,
                      fontsize=9.5,fontweight="bold",zorder=7)
        label_text = _short(node["name"]) + (f" {badge}" if badge else "")
        border_col = node_color if is_special else "#444444"
        name_color = node_color if is_special else TEXT_BRIGHT
        ax_pitch.text(node["avg_x"],node["avg_y"]+9,label_text,
                      ha="center",va="bottom",color=name_color,
                      fontsize=9,fontweight="bold",zorder=6,
                      bbox=dict(boxstyle="round,pad=0.32",facecolor="#000000",
                                alpha=0.82,edgecolor=border_col,lw=1.5))

    tp = events[(events["is_pass"]==True)&(events["team_id"]==team_id)]
    total_p   = len(tp)
    success_p = int((tp["outcome"]=="Successful").sum())
    acc_p     = round(success_p/total_p*100,1) if total_p else 0
    key_p     = int(tp["is_key_pass"].sum())
    top_pairs = sorted(edges.items(),key=lambda x:x[1],reverse=True)[:6]

    stats_lines = [
        ("PASS NETWORK",None,True),(team_name,team_color,True),("",None,False),
        (f"Players:   {len(nodes)}",None,False),
        (f"Passes:    {total_p}",None,False),
        (f"Completed: {success_p}",None,False),
        (f"Accuracy:  {acc_p}%",None,False),
        (f"Key passes:{key_p}",None,False),
        ("",None,False),("TOP PAIRS",None,True),
    ]
    for (pa,pb),cnt in top_pairs:
        na = _short(nodes[pa]["name"]) if pa in nodes else "?"
        nb = _short(nodes[pb]["name"]) if pb in nodes else "?"
        stats_lines.append((f"{na} ↔ {nb}  ({cnt})",None,False))

    y_pos = 0.97
    for text,color,bold in stats_lines:
        if text == "": y_pos -= 0.025; continue
        fc = color if color else TEXT_MAIN
        ax_stats.text(0.08,y_pos,text,transform=ax_stats.transAxes,
                      ha="left",va="top",color=fc,
                      fontsize=10 if bold else 9.5,
                      fontweight="bold" if bold else "normal")
        y_pos -= 0.048 if bold else 0.042

    ax_pitch.legend(handles=[
        Line2D([0],[0],color=C_GOLD, lw=2,alpha=0.6,label="Low connection"),
        Line2D([0],[0],color=C_GREEN,lw=8,alpha=0.85,label="High connection"),
        mpatches.Patch(facecolor=team_color,    edgecolor="white",label="Starter"),
        mpatches.Patch(facecolor=COLOR_SUB_IN,  edgecolor="white",label="Sub In (↑)"),
        mpatches.Patch(facecolor=COLOR_SUB_OUT, edgecolor="white",label="Subbed Off (↓)"),
        mpatches.Patch(facecolor=COLOR_BOTH_SUB,edgecolor="white",label="Sub In+Off (↕)"),
        mpatches.Patch(facecolor=COLOR_RED_CARD,edgecolor="white",label="Red Card (🟥)"),
    ],fontsize=9,ncol=4,facecolor=BG_MID,edgecolor=GRID_COL,
       labelcolor=TEXT_MAIN,loc="lower center",
       bbox_to_anchor=(0.5,-0.08),framealpha=0.92,
       borderpad=0.9,handlelength=2.0)
    # ── Team colour bar ───────────────────────────────────────────
    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0,1); _hdr.set_ylim(0,1); _hdr.axis("off")
    _hdr.add_patch(plt.Rectangle((0,0),1.0,1,facecolor=team_color,alpha=0.93,zorder=0))
    _hdr.add_patch(plt.Rectangle((0,0.82),1.0,0.18,facecolor="white",alpha=0.07,zorder=1))
    _hdr.text(0.015,0.50,f"● {team_name}",ha="left",va="center",
              color="white",fontsize=8.5,fontweight="bold",zorder=3)
    _hdr.text(0.50,0.50,"Created by Mostafa Saad",ha="center",va="center",
              color="#FFD700",fontsize=8,fontweight="bold",fontstyle="italic",zorder=3)
    fig.text(0.50, 0.975, f"Pass Network — {team_name}",
             ha="center", va="top",
             color=TEXT_BRIGHT, fontsize=15, fontweight="bold",
             transform=fig.transFigure,
             path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    fig.text(0.50, 0.951,
             f"Players: {len(nodes)}   Passes: {total_p}   Completed: {success_p} ({acc_p}%)   Key: {key_p}",
             ha="center", va="top",
             color=TEXT_DIM, fontsize=9,
             transform=fig.transFigure)


# ══════════════════════════════════════════════════════
#  FIG 9 & 10 — xT MAP
# ══════════════════════════════════════════════════════
def draw_xt_map_full(fig, events, team_id, team_name, team_color):
    fig.clear(); fig.patch.set_facecolor(BG_DARK)
    gs  = GridSpec(1,2,figure=fig,width_ratios=[4.0,1.1],
                   left=0.05,right=0.96,top=0.87,bottom=0.10,wspace=0.06)
    ax  = fig.add_subplot(gs[0,0])
    axs = fig.add_subplot(gs[0,1])
    axs.set_facecolor(BG_MID); axs.axis("off")

    cmap = LinearSegmentedColormap.from_list("xt",[
        "#0a1628","#0d3b6e","#1a6b3c","#f59e0b","#e63946","#ff0044"])

    grid_display = XT_GRID.T
    rows_n,cols_n = grid_display.shape
    cell_w = 100/cols_n; cell_h = 100/rows_n

    im = ax.imshow(grid_display,extent=[0,100,0,100],origin="lower",
                   aspect="auto",cmap=cmap,vmin=0,vmax=0.76,alpha=0.80,zorder=1)

    x_centers = [(i+0.5)*cell_w for i in range(cols_n)]
    y_centers  = [(j+0.5)*cell_h for j in range(rows_n)]
    for xi,xc in enumerate(x_centers):
        for yi,yc in enumerate(y_centers):
            val = grid_display[yi,xi]
            tc  = "white" if val < 0.28 else "#111111"
            ax.text(xc,yc,f"{val:.3f}",ha="center",va="center",
                    color=tc,fontsize=7.8,fontweight="bold",zorder=3,
                    path_effects=[pe.withStroke(linewidth=1.5,
                        foreground="black" if tc=="white" else "white")])

    cbar = fig.colorbar(im,ax=ax,orientation="vertical",fraction=0.022,pad=0.015)
    cbar.set_label("xT Value",color="white",fontsize=10,labelpad=8)
    cbar.ax.tick_params(colors="white",labelsize=9)
    cbar.outline.set_edgecolor("#334155")

    def pline(xs,ys,lw=1.5,alpha=0.75,ls="-"):
        ax.plot(xs,ys,color="white",lw=lw,alpha=alpha,ls=ls,zorder=4,
                solid_capstyle="round")
    pline([0,100,100,0,0],[0,0,100,100,0],lw=2.0)
    pline([50,50],[0,100],lw=1.2,alpha=0.50,ls="--")
    ax.add_patch(plt.Circle((50,50),9.15*0.68,color="white",fill=False,
                             lw=1.0,alpha=0.40,zorder=4))
    pline([83.5,100,100,83.5,83.5],[21.1,21.1,78.9,78.9,21.1],lw=1.1,alpha=0.50)
    pline([0,16.5,16.5,0,0],        [21.1,21.1,78.9,78.9,21.1],lw=1.1,alpha=0.50)
    pline([94.5,100,100,94.5],       [36.8,36.8,63.2,63.2],lw=0.9,alpha=0.40)
    pline([0,5.5,5.5,0],             [36.8,36.8,63.2,63.2],lw=0.9,alpha=0.40)
    pline([100,100],[44,56],lw=4.0,alpha=0.92)
    pline([0,0],    [44,56],lw=4.0,alpha=0.92)
    ax.axvline(FINAL_THIRD_X,color="white",lw=1.3,ls=":",alpha=0.40,zorder=4)
    ax.axvline(PENALTY_BOX_X,color="white",lw=1.0,ls=":",alpha=0.30,zorder=4)
    for txt,xp,yp,fs in [
        ("Defensive\nThird",25,98,8.5),("Middle\nThird",58,98,8.5),
        ("Final\nThird",83,98,8.5),("Penalty\nBox",91,50,8.0),
        ("← Attack Direction →",50,-5.5,9.5),
    ]:
        ax.text(xp,yp,txt,ha="center",va="top" if yp>5 else "center",
                color="white",fontsize=fs,alpha=0.65,fontweight="bold",zorder=5)
    ax.set_xlim(-2,104); ax.set_ylim(-8,104); ax.axis("off")

    passes = events[
        (events["is_pass"]==True) &
        (events["team_id"]==team_id) &
        (events["outcome"]=="Successful") &
        events[["x","y","end_x","end_y"]].notna().all(axis=1)
    ].copy()
    if "xT" not in passes.columns or passes["xT"].isna().all():
        passes["xT"] = passes.apply(
            lambda r: calc_xt_pass(r["x"],r["y"],r["end_x"],r["end_y"]),axis=1)
    passes = passes[passes["xT"].notna()]
    xt_max = passes["xT"].abs().max() if not passes.empty else 1.0

    for _,row in passes.iterrows():
        val = row["xT"]
        if val >= 0:
            ratio = val/xt_max if xt_max else 0
            col_  = "white"; lw_ = 0.5+ratio*2.6; alpha = 0.12+ratio*0.72
        else:
            ratio = abs(val)/xt_max if xt_max else 0
            col_  = "#ff6b6b"; lw_ = 0.5+ratio*1.2; alpha = 0.10+ratio*0.38
        ax.annotate("",xy=(row["end_x"],row["end_y"]),xytext=(row["x"],row["y"]),
                    arrowprops=dict(arrowstyle="-|>",color=col_,
                                    lw=lw_,alpha=alpha,mutation_scale=6),zorder=6)

    if not passes.empty:
        top5 = passes.nlargest(5,"xT"); used = []
        for rank,(_,row) in enumerate(top5.iterrows(),1):
            ax.annotate("",xy=(row["end_x"],row["end_y"]),
                        xytext=(row["x"],row["y"]),
                        arrowprops=dict(arrowstyle="-|>",color="#facc15",
                                        lw=5.5,alpha=0.15,mutation_scale=14),zorder=7)
            ax.annotate("",xy=(row["end_x"],row["end_y"]),
                        xytext=(row["x"],row["y"]),
                        arrowprops=dict(arrowstyle="-|>",color="#facc15",
                                        lw=2.8,alpha=0.96,mutation_scale=12),zorder=8)
            ax.scatter(row["x"],row["y"],c="#facc15",s=45,
                       edgecolors="black",lw=0.8,zorder=9)
            ax.scatter(row["end_x"],row["end_y"],c="#facc15",s=180,
                       marker="*",edgecolors="white",lw=1.2,zorder=9)
            lbl_y = row["end_y"]+4.5
            for px,py in used:
                if abs(row["end_x"]-px)<12 and abs(lbl_y-py)<6: lbl_y += 6
            used.append((row["end_x"],lbl_y))
            ax.text(row["end_x"],lbl_y,
                    f"#{rank} {_short(row.get('player',''))}  +{row['xT']:.3f}",
                    ha="center",va="bottom",color="#facc15",fontsize=8.2,
                    fontweight="bold",zorder=10,
                    bbox=dict(boxstyle="round,pad=0.30",facecolor="#050508",
                              alpha=0.85,edgecolor="#facc15",lw=1.0))

    xt_total = round(passes["xT"].sum(),3)  if not passes.empty else 0.0
    xt_best  = round(passes["xT"].max(),3)  if not passes.empty else 0.0
    n_pos    = int((passes["xT"]>0).sum()) if not passes.empty else 0
    n_neg    = int((passes["xT"]<0).sum()) if not passes.empty else 0
    n_total  = len(passes)
    axs.set_xlim(0,1); axs.set_ylim(0,1)

    def card(y0,height,color,alpha=0.18):
        axs.add_patch(mpatches.FancyBboxPatch(
            (0.05,y0),0.90,height,boxstyle="round,pad=0.01",
            facecolor=color,edgecolor=color,alpha=alpha,lw=1.2,
            transform=axs.transAxes,zorder=1))
    def divider(y):
        axs.plot([0.05,0.95],[y,y],color="#2d3748",lw=1.0,
                 transform=axs.transAxes,zorder=2)
    def lbl(txt,x,y,color=TEXT_MAIN,size=9,bold=False,ha="left"):
        axs.text(x,y,txt,transform=axs.transAxes,ha=ha,va="center",color=color,
                 fontsize=size,fontweight="bold" if bold else "normal",
                 clip_on=False,zorder=3)

    card(0.91,0.08,team_color,alpha=0.25)
    lbl("⚡  xT MAP",0.50,0.96,team_color,size=11,bold=True,ha="center")
    lbl(team_name,  0.50,0.92,team_color,size=9, bold=True,ha="center")
    divider(0.90)
    card(0.78,0.11,team_color,alpha=0.20)
    lbl("TOTAL  xT",        0.50,0.875,TEXT_DIM,  size=8, ha="center")
    lbl(f"{xt_total:+.4f}",0.50,0.820,TEXT_BRIGHT,size=16,bold=True,ha="center")
    divider(0.77)
    yq = 0.74
    for label,val,col in [
        ("Best Pass",   f"+{xt_best:.3f}",TEXT_MAIN),
        ("Total Passes",str(n_total),     TEXT_MAIN),
        ("Positive xT", str(n_pos),       "#22c55e"),
        ("Negative xT", str(n_neg),       "#e63946"),
    ]:
        lbl(label,0.08,yq,TEXT_DIM,size=8.5)
        lbl(val,  0.92,yq,col,size=9,bold=True,ha="right")
        yq -= 0.052
    divider(yq+0.010); yq -= 0.010
    lbl("TOP  xT  CREATORS",0.50,yq,TEXT_DIM,size=8.5,bold=True,ha="center")
    yq -= 0.045
    if not passes.empty:
        top_p = (passes[passes["xT"]>0].groupby("player")["xT"]
                 .sum().sort_values(ascending=False).head(7))
        mx = top_p.max() if not top_p.empty else 1.0
        for rank,(player,xval) in enumerate(top_p.items(),1):
            bw  = max(min(xval/mx*0.78,0.78),0.04)
            bar_col = team_color if rank>1 else "#facc15"
            axs.add_patch(mpatches.FancyBboxPatch(
                (0.08,yq-0.018),0.84,0.036,boxstyle="round,pad=0.005",
                facecolor="#1e2836",edgecolor="none",
                transform=axs.transAxes,zorder=2))
            axs.add_patch(mpatches.FancyBboxPatch(
                (0.08,yq-0.018),bw*0.84,0.036,boxstyle="round,pad=0.005",
                facecolor=bar_col,edgecolor="none",alpha=0.45,
                transform=axs.transAxes,zorder=3))
            name_col = "#facc15" if rank==1 else TEXT_BRIGHT
            lbl(f"{rank}. {_short(player)}",0.10,yq,name_col,size=8.2,bold=(rank==1))
            lbl(f"+{xval:.3f}",0.90,yq,"#facc15" if rank==1 else TEXT_DIM,
                size=8.2,bold=(rank==1),ha="right")
            yq -= 0.060

    ax.legend(handles=[
        mpatches.Patch(facecolor="white",  alpha=0.80,label="Positive xT pass"),
        mpatches.Patch(facecolor="#ff6b6b",alpha=0.80,label="Negative xT pass"),
        mpatches.Patch(facecolor="#facc15",edgecolor="white",lw=1,
                       label="⭐ Top-5 xT passes"),
    ],fontsize=9.5,ncol=3,facecolor=BG_MID,edgecolor=GRID_COL,
       labelcolor=TEXT_MAIN,loc="lower center",
       bbox_to_anchor=(0.5,-0.05),framealpha=0.92)
    # ── Team colour bar ───────────────────────────────────────────
    _hdr = fig.add_axes([0.0, 0.980, 1.0, 0.020])
    _hdr.set_xlim(0,1); _hdr.set_ylim(0,1); _hdr.axis("off")
    _hdr.add_patch(plt.Rectangle((0,0),1.0,1,facecolor=team_color,alpha=0.93,zorder=0))
    _hdr.add_patch(plt.Rectangle((0,0.82),1.0,0.18,facecolor="white",alpha=0.07,zorder=1))
    _hdr.text(0.015,0.50,f"● {team_name}",ha="left",va="center",
              color="white",fontsize=8.5,fontweight="bold",zorder=3)
    _hdr.text(0.50,0.50,"Created by Mostafa Saad",ha="center",va="center",
              color="#FFD700",fontsize=8,fontweight="bold",fontstyle="italic",zorder=3)
    fig.text(0.50, 0.975, f"xT Map — {team_name}",
             ha="center", va="top",
             color=TEXT_BRIGHT, fontsize=15, fontweight="bold",
             transform=fig.transFigure,
             path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    fig.text(0.50, 0.951,
             f"Total xT: {xt_total:+.3f}   ⭐ Best Pass: +{xt_best:.3f}   Passes: {n_total}",
             ha="center", va="top",
             color=TEXT_DIM, fontsize=9,
             transform=fig.transFigure)


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
        ax.plot(xs, ys, color="white", lw=lw, alpha=a, ls=ls,
                zorder=2, solid_capstyle="round")
    # ── Subtle box fills ──────────────────────────────────────────
    ax.add_patch(plt.Rectangle((83.5, 21.1), 16.5, 57.8,
                 facecolor="#ffffff", alpha=0.025, edgecolor="none", zorder=1))
    ax.add_patch(plt.Rectangle((0, 21.1), 16.5, 57.8,
                 facecolor="#ffffff", alpha=0.018, edgecolor="none", zorder=1))
    # ── Pitch outline & markings ──────────────────────────────────
    l([0,100,100,0,0], [0,0,100,100,0], lw=1.9, a=0.88)
    l([50,50], [0,100], lw=1.0, a=0.40, ls="--")
    ax.add_patch(plt.Circle((50,50), 9.15*0.68, color="white", fill=False,
                             lw=0.9, alpha=0.38, zorder=2))
    ax.plot(50, 50, "o", color="white", ms=2.0, alpha=0.55, zorder=2)
    l([83.5,100,100,83.5,83.5], [21.1,21.1,78.9,78.9,21.1], lw=1.0, a=0.55)
    l([0,16.5,16.5,0,0],        [21.1,21.1,78.9,78.9,21.1], lw=1.0, a=0.55)
    l([94.5,100,100,94.5],       [36.8,36.8,63.2,63.2],      lw=0.75, a=0.42)
    l([0,5.5,5.5,0],             [36.8,36.8,63.2,63.2],      lw=0.75, a=0.42)
    # Goal posts — brighter
    l([100,100],[44,56], lw=4.0, a=0.96)
    l([0,0],   [44,56], lw=4.0, a=0.96)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-8, 104)
    ax.axis("off")


def _lbl(ax, txt, col=TEXT_BRIGHT, size=8.5):
    """
    Title above any axes — uses transAxes so it works for ALL panel types.
    Enhanced with glow path-effect and sharper badge.
    """
    ax.text(
        0.50, 1.028, txt,
        ha="center", va="bottom",
        transform=ax.transAxes,
        color=col, fontsize=size, fontweight="bold",
        zorder=20, clip_on=False,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="#000000")],
        bbox=dict(
            boxstyle="round,pad=0.40",
            facecolor="#07090f",
            edgecolor=col,
            linewidth=1.4,
            alpha=0.97,
        ),
    )

def _rpt_pass_network(ax,events,tid,tc,name):
    _mini_pitch(ax); _lbl(ax,f"Pass Network — {name}",tc)
    p = events[(events["is_pass"]==True)&(events["team_id"]==tid)&
               (events["outcome"]=="Successful")&
               events[["x","y","end_x","end_y"]].notna().all(axis=1)].copy()
    if p.empty: return
    avg = p.groupby("player")[["x","y"]].mean()
    cnt = p.groupby("player").size().rename("n")
    avg = avg.join(cnt)
    mx_n = avg["n"].max() if not avg.empty else 1
    for pl,row in avg.iterrows():
        s = 18+row["n"]/mx_n*85
        ax.scatter(row["x"],row["y"],c=tc,s=s,edgecolors="white",lw=0.7,zorder=5)
        nm = pl.split()[-1][:7] if pl else ""
        ax.text(row["x"],row["y"]+3.5,nm,ha="center",va="bottom",
                color="white",fontsize=5.5,zorder=6,
                bbox=dict(boxstyle="round,pad=0.15",facecolor="#000000",
                          alpha=0.70,edgecolor="none"))

def _rpt_shot_table(ax,events,info,xg_data):
    ax.set_facecolor(BG_MID); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    hn,an = info["home_name"],info["away_name"]
    hd,ad = xg_data.get(hn,{}),xg_data.get(an,{})
    hs  = events[(events["is_shot"]==True)&(events["team_id"]==info["home_id"])]
    as_ = events[(events["is_shot"]==True)&(events["team_id"]==info["away_id"])]
    hxgot = round(hs[hs["shot_category"].isin(["Saved","Goal"])]["xG"].sum(),2)
    axgot = round(as_[as_["shot_category"].isin(["Saved","Goal"])]["xG"].sum(),2)
    rows_=[("Goals",  hd.get("goals",0),      ad.get("goals",0),      C_GOLD),
           ("xG",     hd.get("xG",0),         ad.get("xG",0),         "#a855f7"),
           ("xGoT",   hxgot,                  axgot,                  "#1e90ff"),
           ("Shots",  hd.get("shots",0),       ad.get("shots",0),      "#94a3b8"),
           ("On Tgt", hd.get("on_target",0),   ad.get("on_target",0),  C_GREEN),
           ("Blocked",hd.get("blocked",0),     ad.get("blocked",0),    C_GOLD),
           ("Missed", hd.get("missed",0),      ad.get("missed",0),     "#64748b"),
           ("Big Ch.",hd.get("big_chances",0), ad.get("big_chances",0),"#f43f5e")]
    ax.text(0.10,0.97,hn[:10],ha="left",va="top",color=C_RED,fontsize=8.5,
            fontweight="bold",transform=ax.transAxes)
    ax.text(0.50,0.97,"SHOTS",ha="center",va="top",color=TEXT_DIM,fontsize=7.5,
            fontweight="bold",transform=ax.transAxes)
    ax.text(0.90,0.97,an[:10],ha="right",va="top",color=C_BLUE,fontsize=8.5,
            fontweight="bold",transform=ax.transAxes)
    ax.plot([0.03,0.97],[0.93,0.93],color=GRID_COL,lw=0.8,transform=ax.transAxes)
    y=0.875
    for lbl_,hv,av,col in rows_:
        tot=(float(hv)+float(av)) or 1; hr=float(hv)/tot
        ax.barh(y,hr*0.22,height=0.055,left=0.03,color=C_RED,alpha=0.55,
                transform=ax.transAxes,zorder=2)
        ax.barh(y,(1-hr)*0.22,height=0.055,left=0.75,color=C_BLUE,alpha=0.55,
                transform=ax.transAxes,zorder=2)
        ax.text(0.27,y,str(hv),ha="right",va="center",color=C_RED,fontsize=10,
                fontweight="bold",transform=ax.transAxes)
        ax.text(0.50,y,lbl_,ha="center",va="center",color=col,fontsize=7.5,
                fontweight="bold",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.20",facecolor=BG_DARK,
                          alpha=0.85,edgecolor=col,lw=0.8))
        ax.text(0.73,y,str(av),ha="left",va="center",color=C_BLUE,fontsize=10,
                fontweight="bold",transform=ax.transAxes)
        y -= 0.102
    _lbl(ax,"Shot Statistics",TEXT_DIM)

def _rpt_avg_position(ax,events,tid,tc,name):
    _mini_pitch(ax); _lbl(ax,f"Avg Positions — {name}",tc)
    ev = events[(events["team_id"]==tid)&events[["x","y"]].notna().all(axis=1)].copy()
    if ev.empty: return
    avg = ev.groupby("player")[["x","y"]].mean()
    cnt = ev.groupby("player").size()
    mx  = cnt.max() if not cnt.empty else 1
    for pl,row in avg.iterrows():
        n=cnt.get(pl,1)
        ax.scatter(row["x"],row["y"],c=tc,s=25+n/mx*80,
                   edgecolors="white",lw=0.7,alpha=0.85,zorder=4)
        nm=pl.split()[-1][:6] if pl else ""
        ax.text(row["x"],row["y"]+3.2,nm,ha="center",va="bottom",
                color="white",fontsize=5.2,zorder=5,
                bbox=dict(boxstyle="round,pad=0.12",facecolor="#000000",
                          alpha=0.68,edgecolor="none"))

def _rpt_gk_saves(ax,events,info):
    ax.set_facecolor(BG_DARK); ax.axis("off")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    _lbl(ax,"Goalkeeper Saves",TEXT_DIM)
    def goal_box(cx,cy,w,h,col,lbl_,n):
        ax.plot([cx-w/2,cx+w/2,cx+w/2,cx-w/2,cx-w/2],
                [cy,cy,cy+h,cy+h,cy],color="white",lw=1.5,alpha=0.8,zorder=3)
        ax.text(cx,cy-6,f"{lbl_[:9]}\n{n} saves",
                ha="center",va="top",color=col,fontsize=7,fontweight="bold")
    goal_box(25,30,36,30,C_RED, info["home_name"],
             len(events[(events["team_id"]==info["away_id"]) &
                         (events["shot_category"]=="Saved")]))
    goal_box(75,30,36,30,C_BLUE,info["away_name"],
             len(events[(events["team_id"]==info["home_id"]) &
                         (events["shot_category"]=="Saved")]))
    for tid,cx,col in [(info["away_id"],25,C_RED),(info["home_id"],75,C_BLUE)]:
        sv = events[(events["team_id"]==tid) &
                    (events["shot_category"]=="Saved") &
                    events[["end_x","end_y"]].notna().all(axis=1)]
        if sv.empty: continue
        sx = cx+(sv["end_y"]-50)*0.36
        sy = 30+(sv["end_x"].clip(83.5,100)-83.5)/16.5*30
        ax.scatter(sx,sy,c=col,s=55,edgecolors="white",lw=0.8,alpha=0.88,zorder=5)

def _rpt_progressive(ax,events,tid,tc,name):
    _mini_pitch(ax)
    p    = events[(events["is_pass"]==True)&(events["team_id"]==tid)&
                  (events["outcome"]=="Successful")&
                  events[["x","y","end_x","end_y"]].notna().all(axis=1)].copy()
    prog = p[(p["end_x"]-p["x"])>=10].copy() if not p.empty else p
    n    = len(prog)
    _lbl(ax,f"{name}: {n} Progressive Passes",tc)
    if prog.empty: return
    def zc(x):
        if x<33: return "#64748b"
        if x<66: return C_GOLD
        return C_GREEN
    for _,r in prog.iterrows():
        ax.annotate("",xy=(r["end_x"],r["end_y"]),xytext=(r["x"],r["y"]),
                    arrowprops=dict(arrowstyle="-|>",color=zc(r["x"]),
                                    lw=0.9,alpha=0.55,mutation_scale=5),zorder=3)
    n1=int((prog["x"]<33).sum()); n2=int(((prog["x"]>=33)&(prog["x"]<66)).sum())
    n3=int((prog["x"]>=66).sum())
    for i,(lv,cl) in enumerate([
        (f"Own 3rd  {n1}({int(n1/n*100) if n else 0}%)","#64748b"),
        (f"Middle   {n2}({int(n2/n*100) if n else 0}%)",C_GOLD),
        (f"Final 3rd {n3}({int(n3/n*100) if n else 0}%)",C_GREEN),
    ]):
        ax.text(1,99-i*9,lv,ha="right",va="top",color=cl,fontsize=6.5,
                fontweight="bold",zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",facecolor=BG_DARK,
                          alpha=0.80,edgecolor="none"))

def _rpt_xt_minute(ax,events,info):
    ax.set_facecolor(BG_DARK)
    _lbl(ax,"xT — Match Dominance per Minute",TEXT_DIM)
    if "xT" not in events.columns:
        ax.text(0.5,0.5,"No xT data",ha="center",va="center",
                color=TEXT_DIM,transform=ax.transAxes); return
    xt  = events[(events["xT"].notna())&(events["xT"]>0)&
                 (events["outcome"]=="Successful")].copy()
    hxt = xt[xt["team_id"]==info["home_id"]].groupby("minute")["xT"].sum()
    axt = xt[xt["team_id"]==info["away_id"]].groupby("minute")["xT"].sum()
    mins = list(range(1,96))
    ax.bar(mins,[hxt.get(m,0) for m in mins],color=C_RED,alpha=0.75,width=0.8)
    ax.bar(mins,[-axt.get(m,0) for m in mins],color=C_BLUE,alpha=0.75,width=0.8)
    ax.axhline(0,color="white",lw=0.8,alpha=0.5)
    for xp,lb in [(45,"HT"),(90,"FT")]:
        ax.axvline(xp,color=C_GOLD,lw=1.0,ls="--",alpha=0.60)
        ax.text(xp+0.5,ax.get_ylim()[1]*0.85,lb,color=C_GOLD,fontsize=6)
    ht=round(xt[xt["team_id"]==info["home_id"]]["xT"].sum(),3)
    at=round(xt[xt["team_id"]==info["away_id"]]["xT"].sum(),3)
    for pos,name,col,ha_ in [
        (0.02,info["home_name"],C_RED,"left"),
        (0.98,info["away_name"],C_BLUE,"right"),
    ]:
        ax.text(pos,0.97,f"{name[:10]}  xT:{ht if col==C_RED else at}",
                transform=ax.transAxes,ha=ha_,va="top",color=col,fontsize=7.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22",facecolor=BG_MID,
                          alpha=0.85,edgecolor="none"))
    ax.tick_params(colors=TEXT_DIM,labelsize=7)
    ax.set_xlabel("Minute",color=TEXT_DIM,fontsize=7,labelpad=2)
    ax.set_ylabel("xT / min",color=TEXT_DIM,fontsize=7,labelpad=2)
    for sp in ["top","right"]:   ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GRID_COL)
    ax.grid(alpha=0.07,color=GRID_COL)

def _rpt_zone14(ax,events,tid,tc,name):
    _mini_pitch(ax); _lbl(ax,f"Zone 14 & Half-Spaces — {name}",tc)
    ev=events[(events["team_id"]==tid)&events[["x","y"]].notna().all(axis=1)].copy()
    from matplotlib.patches import Rectangle as Rect
    ax.add_patch(Rect((66,33),17,34,facecolor=tc,alpha=0.22,edgecolor=tc,lw=1.5,zorder=2))
    ax.add_patch(Rect((66,67),17,13,facecolor="#a855f7",alpha=0.22,
                      edgecolor="#a855f7",lw=1.2,zorder=2))
    ax.add_patch(Rect((66,20),17,13,facecolor="#a855f7",alpha=0.22,
                      edgecolor="#a855f7",lw=1.2,zorder=2))
    if ev.empty: return
    z14=ev[ev["x"].between(66,83)&ev["y"].between(33,67)]
    lhs=ev[ev["x"].between(66,83)&ev["y"].between(67,80)]
    rhs=ev[ev["x"].between(66,83)&ev["y"].between(20,33)]
    if not z14.empty:
        ax.scatter(z14["x"],z14["y"],c=tc,s=12,alpha=0.60,zorder=4,edgecolors="none")
    for val,yx,yy,col in [
        (len(z14),74.5,50,tc),(len(lhs),74.5,73.5,"#a855f7"),
        (len(rhs),74.5,26.5,"#a855f7"),
    ]:
        ax.text(yx,yy,str(val),ha="center",va="center",color="white",
                fontsize=10,fontweight="bold",zorder=6,
                bbox=dict(boxstyle="circle,pad=0.35",facecolor=col,
                          alpha=0.88,edgecolor="white",lw=0.9))
    ax.text(50,-4.5,
            f"Zone14:{len(z14)}  L.Half-Space:{len(lhs)}  R.Half-Space:{len(rhs)}",
            ha="center",va="top",color=tc,fontsize=6.5,fontweight="bold")

def _rpt_stats_table(ax,events,info,xg_data):
    ax.set_facecolor(BG_DARK); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    hn,an = info["home_name"],info["away_name"]
    hid,aid = info["home_id"],info["away_id"]
    hd,ad = xg_data.get(hn,{}),xg_data.get(an,{})
    hp  = int(events[(events["is_pass"]==True)&(events["team_id"]==hid)].shape[0])
    ap  = int(events[(events["is_pass"]==True)&(events["team_id"]==aid)].shape[0])
    tot_p = (hp+ap) or 1
    hps = int(events[(events["is_pass"]==True)&(events["team_id"]==hid)&
                      (events["outcome"]=="Successful")].shape[0])
    aps = int(events[(events["is_pass"]==True)&(events["team_id"]==aid)&
                      (events["outcome"]=="Successful")].shape[0])
    hkp = int(events[(events["is_key_pass"]==True)&(events["team_id"]==hid)].shape[0])
    akp = int(events[(events["is_key_pass"]==True)&(events["team_id"]==aid)].shape[0])
    if "xT" in events.columns:
        hxt = round(events[(events["team_id"]==hid)&events["xT"].notna()]["xT"].sum(),2)
        axt = round(events[(events["team_id"]==aid)&events["xT"].notna()]["xT"].sum(),2)
    else:
        hxt, axt = 0, 0
    stats=[
        ("Possession", f"{round(hp/tot_p*100,1)}%",  f"{round(ap/tot_p*100,1)}%",
         hp/tot_p, C_GOLD),
        ("Passes (Acc)",f"{hp}({hps})",f"{ap}({aps})",
         hp/((hp+ap) or 1),"#94a3b8"),
        ("Shots (OnTgt)",
         f"{hd.get('shots',0)}({hd.get('on_target',0)})",
         f"{ad.get('shots',0)}({ad.get('on_target',0)})",
         hd.get("shots",0)/((hd.get("shots",0)+ad.get("shots",0)) or 1),"#64748b"),
        ("xG", str(hd.get("xG",0)),str(ad.get("xG",0)),
         hd.get("xG",0)/((hd.get("xG",0)+ad.get("xG",0)) or 1),"#a855f7"),
        ("xT", str(hxt),str(axt),hxt/((hxt+axt) or 1),"#22c55e"),
        ("Key Passes",str(hkp),str(akp),hkp/((hkp+akp) or 1),"#1e90ff"),
        ("Big Chances",str(hd.get("big_chances",0)),str(ad.get("big_chances",0)),
         hd.get("big_chances",0)/((hd.get("big_chances",0)+ad.get("big_chances",0)) or 1),
         "#f43f5e"),
    ]
    ax.text(0.50,0.99,"MATCH STATISTICS",ha="center",va="top",color=TEXT_BRIGHT,
            fontsize=9.5,fontweight="bold",transform=ax.transAxes)
    ax.text(0.10,0.94,hn[:12],ha="left",va="top",color=C_RED,fontsize=8,
            fontweight="bold",transform=ax.transAxes)
    ax.text(0.90,0.94,an[:12],ha="right",va="top",color=C_BLUE,fontsize=8,
            fontweight="bold",transform=ax.transAxes)
    ax.plot([0.02,0.98],[0.90,0.90],color=GRID_COL,lw=0.8,transform=ax.transAxes)
    y=0.85; step=0.118
    for lbl_,hv,av,hr,col in stats:
        ax.add_patch(plt.Rectangle((0.02,y-0.048),hr*0.96,0.050,
                     facecolor=C_RED,alpha=0.18,transform=ax.transAxes,zorder=1))
        ax.add_patch(plt.Rectangle((0.02+hr*0.96,y-0.048),(1-hr)*0.96,0.050,
                     facecolor=C_BLUE,alpha=0.18,transform=ax.transAxes,zorder=1))
        ax.text(0.06,y,str(hv),ha="left",va="center",color=C_RED,fontsize=9,
                fontweight="bold",transform=ax.transAxes)
        ax.text(0.50,y,lbl_,ha="center",va="center",color=col,fontsize=7.8,
                fontweight="bold",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.22",facecolor=BG_MID,
                          alpha=0.92,edgecolor=col,lw=0.8))
        ax.text(0.94,y,str(av),ha="right",va="center",color=C_BLUE,fontsize=9,
                fontweight="bold",transform=ax.transAxes)
        y -= step

def _rpt_pass_zones(ax,events,tid,tc,name):
    _mini_pitch(ax); _lbl(ax,f"Pass Zones — {name}",tc)
    p=events[(events["is_pass"]==True)&(events["team_id"]==tid)&
             events[["x","y"]].notna().all(axis=1)].copy()
    if p.empty: return
    tot=len(p)
    col_e=[0,20,40,60,80,100]; row_e=[0,34,67,100]
    for ci in range(5):
        for ri in range(3):
            x0,x1=col_e[ci],col_e[ci+1]; y0,y1=row_e[ri],row_e[ri+1]
            n=int(p[(p["x"]>=x0)&(p["x"]<x1)&
                    (p["y"]>=y0)&(p["y"]<y1)].shape[0])
            if n==0: continue
            pct=round(n/tot*100,0)
            inten=min(pct/12,1.0)
            ax.add_patch(plt.Rectangle((x0+0.5,y0+0.5),x1-x0-1,y1-y0-1,
                         facecolor=tc,alpha=inten*0.58,edgecolor="none",zorder=2))
            ax.text((x0+x1)/2,(y0+y1)/2,f"{int(pct)}%",
                    ha="center",va="center",color="white",fontsize=7.5,
                    fontweight="bold",zorder=3,
                    path_effects=[pe.withStroke(linewidth=1.8,foreground="black")])

def _rpt_crosses(ax,events,info):
    _mini_pitch(ax); _lbl(ax,"Crosses",TEXT_DIM)
    if "is_cross" not in events.columns:
        ax.text(50,50,"No cross data",ha="center",va="center",
                color=TEXT_DIM,fontsize=8,style="italic"); return
    crs = events[
        (events["is_cross"]==True) &
        events[["x","y","end_x","end_y"]].notna().all(axis=1)
    ].copy()
    if crs.empty:
        ax.text(50,50,"No crosses recorded",ha="center",va="center",
                color=TEXT_DIM,fontsize=8,style="italic"); return
    for tid,col in [(info["home_id"],C_RED),(info["away_id"],C_BLUE)]:
        tc=crs[crs["team_id"]==tid]
        for _,r in tc.iterrows():
            succ=r.get("outcome","")=="Successful"
            ax.annotate("",xy=(r["end_x"],r["end_y"]),xytext=(r["x"],r["y"]),
                        arrowprops=dict(arrowstyle="-|>",color=col,
                                        lw=1.3 if succ else 0.6,
                                        alpha=0.82 if succ else 0.30,
                                        mutation_scale=6),zorder=4)
        eff=int(tc[tc["outcome"]=="Successful"].shape[0])
        nm = info["home_name"][:8] if tid==info["home_id"] else info["away_name"][:8]
        xp = 2 if tid==info["home_id"] else 98
        ax.text(xp,-4.5,f"{nm}: {len(tc)} ({eff} eff.)",
                ha="left" if tid==info["home_id"] else "right",
                va="top",color=col,fontsize=6.5,fontweight="bold")

def _rpt_danger_zones(ax,events,tid,tc,name):
    _mini_pitch(ax); _lbl(ax,f"Danger Creation — {name}",tc)
    dng=events[((events["is_shot"]==True)|(events["is_key_pass"]==True))&
               (events["team_id"]==tid)&
               events[["x","y"]].notna().all(axis=1)].copy()
    if dng.empty: return
    shots = dng[dng["is_shot"]==True]
    kp    = dng[dng["is_key_pass"]==True]
    goals = shots[shots["is_goal"]==True] if not shots.empty else shots
    if not dng.empty:
        ax.scatter(dng["x"],dng["y"],c=tc,s=120,alpha=0.08,zorder=2,edgecolors="none")
        ax.scatter(dng["x"],dng["y"],c=tc,s=40, alpha=0.12,zorder=2,edgecolors="none")
    if not kp.empty:
        ax.scatter(kp["x"],kp["y"],c="#facc15",s=22,marker="^",
                   alpha=0.78,zorder=4,edgecolors="none")
    if not shots.empty:
        ax.scatter(shots["x"],shots["y"],c=tc,s=28,alpha=0.75,
                   zorder=4,edgecolors="white",lw=0.5)
    if not goals.empty:
        ax.scatter(goals["x"],goals["y"],c="#FFD700",s=90,marker="*",
                   zorder=6,edgecolors="white",lw=0.8)
    ax.text(50,-4.5,
            f"Shots:{len(shots)}  Key Pass:{len(kp)}  Goals:{len(goals)}",
            ha="center",va="top",color=tc,fontsize=6.5,fontweight="bold")
    ax.legend(handles=[
        Line2D([0],[0],marker="^",color="w",markerfacecolor="#facc15",
               markersize=6,linestyle="None",label="Key Pass"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=tc,
               markersize=6,linestyle="None",label="Shot"),
        Line2D([0],[0],marker="*",color="w",markerfacecolor="#FFD700",
               markersize=8,linestyle="None",label="Goal"),
    ],fontsize=5.5,facecolor=BG_MID,edgecolor="none",
      labelcolor="white",loc="upper left",markerscale=0.9,framealpha=0.80)

# ═══════════════════════════════════════════════════════════════════
#  WATERMARK & PAGE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

CREDIT_MAIN  = "Created by Mostafa Saad"
CREDIT_TOOLS = "Data: WhoScored  |  xG: Opta-style model  |  xT: Karun Singh"

def _watermark(fig):
    """
    Bottom credit bar — separator + data sources.
    (Creator credit now lives in the top colour bar of each figure.)
    """
    # thin separator line
    fig.add_artist(
        plt.Line2D([0.03, 0.97], [0.026, 0.026],
                   transform=fig.transFigure,
                   color="#1e3a5f", lw=0.8, alpha=0.85)
    )
    # Tool credits only — dim, italic
    fig.text(
        0.50, 0.013, CREDIT_TOOLS,
        ha="center", va="center",
        color="#475569", fontsize=6.5, fontstyle="italic",
        transform=fig.transFigure
    )


def _page_header(fig, hn, an, hg, ag, hxg, axg,
                 hform, aform, venue, status, page_num, page_title):
    from matplotlib.patches import FancyBboxPatch
    fig.patch.set_facecolor(BG_DARK)

    # coloured title bar
    ax = fig.add_axes([0.0, 0.968, 1.0, 0.032])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0,0),   0.470,1, boxstyle="square,pad=0",
                                facecolor=C_RED, alpha=0.90, zorder=0))
    ax.add_patch(FancyBboxPatch((0.530,0),0.470,1, boxstyle="square,pad=0",
                                facecolor=C_BLUE, alpha=0.90, zorder=0))
    ax.add_patch(FancyBboxPatch((0.415,0),0.170,1, boxstyle="square,pad=0",
                                facecolor=BG_DARK, alpha=0.95, zorder=1))
    ax.text(0.235, 0.52, hn, ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(0.765, 0.52, an, ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(0.500, 0.52, f"{hg}  —  {ag}", ha="center", va="center",
            color="#FFD700", fontsize=20, fontweight="bold")

    # info bar
    ax2 = fig.add_axes([0.0, 0.952, 1.0, 0.016])
    ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.axis("off")
    ax2.set_facecolor("#07090f")
    sl, sc = STATUS_BADGE.get(status, ("■ Full Time","#64748b"))
    ax2.text(0.012, 0.50, f"xG: {hxg}  •  {hform}",
             ha="left", va="center", color=C_RED, fontsize=8.5, fontweight="bold")
    ax2.text(0.500, 0.50,
             f"{sl}   •   {venue}   •   {page_title}   •   Page {page_num}/2",
             ha="center", va="center", color=TEXT_DIM, fontsize=8)
    ax2.text(0.988, 0.50, f"{aform}  •  xG: {axg}",
             ha="right", va="center", color=C_BLUE, fontsize=8.5, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════
#  PANEL HELPERS  (each = one team, one metric, independent)
# ═══════════════════════════════════════════════════════════════════

# ── A: Shot Comparison (full-width dual tiles) ─────────────────────
def _panel_shot_comparison(ax, events, info, xg_data):
    ax.set_facecolor(BG_MID); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    hn,an = info["home_name"], info["away_name"]
    hd = xg_data.get(hn,{}); ad = xg_data.get(an,{})
    metrics = [
        ("Goals",       hd.get("goals",0),       ad.get("goals",0),       C_GOLD),
        ("xG",          hd.get("xG",0),           ad.get("xG",0),          "#a855f7"),
        ("Shots",       hd.get("shots",0),        ad.get("shots",0),       "#94a3b8"),
        ("On Target",   hd.get("on_target",0),    ad.get("on_target",0),   C_GREEN),
        ("Blocked",     hd.get("blocked",0),      ad.get("blocked",0),     C_GOLD),
        ("Big Ch.",     hd.get("big_chances",0),  ad.get("big_chances",0), "#f43f5e"),
        ("Missed",      hd.get("missed",0),       ad.get("missed",0),      "#64748b"),
    ]
    n = len(metrics); cw = 1.0/n

    # ── عنوان مع مسافة كافية ───────────────────────────────────────
    ax.text(0.50, 0.975, "SHOT COMPARISON",
            ha="center", va="top", color=TEXT_BRIGHT,
            fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.975, hn[:14],
            ha="left", va="top", color=C_RED,
            fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.98, 0.975, an[:14],
            ha="right", va="top", color=C_BLUE,
            fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    # خط فاصل أسفل العنوان
    ax.plot([0.01, 0.99], [0.91, 0.91], color="#1f2937", lw=0.8,
            transform=ax.transAxes)

    for i,(lbl,hv,av,col) in enumerate(metrics):
        cx  = (i + 0.5) * cw
        tot = (float(hv) + float(av)) or 1
        hr  = float(hv) / tot

        # ── بطاقة الخلفية ─────────────────────────────────────────
        ax.add_patch(mpatches.FancyBboxPatch(
            (i*cw + 0.006, 0.05), cw - 0.012, 0.84,
            boxstyle="round,pad=0.006",
            transform=ax.transAxes,
            facecolor="#060d1a", edgecolor=col, lw=1.3, alpha=0.90))

        # ── شريط النسبة ────────────────────────────────────────────
        BAR_Y, BAR_H = 0.35, 0.18
        ax.add_patch(mpatches.FancyBboxPatch(
            (i*cw + 0.009, BAR_Y), (cw - 0.018) * hr, BAR_H,
            boxstyle="round,pad=0.002",
            transform=ax.transAxes, facecolor=C_RED, alpha=0.88))
        ax.add_patch(mpatches.FancyBboxPatch(
            (i*cw + 0.009 + (cw - 0.018)*hr, BAR_Y),
            (cw - 0.018) * (1 - hr), BAR_H,
            boxstyle="round,pad=0.002",
            transform=ax.transAxes, facecolor=C_BLUE, alpha=0.88))

        # ── الأرقام (كبيرة وواضحة) ─────────────────────────────────
        hv_str = f"{hv:.2f}" if isinstance(hv, float) else str(hv)
        av_str = f"{av:.2f}" if isinstance(av, float) else str(av)
        # Home
        ax.text(cx - cw*0.22, 0.66, hv_str,
                ha="center", va="center", color=C_RED,
                fontsize=15, fontweight="bold",
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground="#000")])
        # Away
        ax.text(cx + cw*0.22, 0.66, av_str,
                ha="center", va="center", color=C_BLUE,
                fontsize=15, fontweight="bold",
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground="#000")])

        # ── الفاصل بين الرقمين ─────────────────────────────────────
        ax.text(cx, 0.66, "–",
                ha="center", va="center", color="#475569",
                fontsize=11, transform=ax.transAxes)

        # ── اسم المقياس (مع مسافة كافية عن الشريط) ────────────────
        ax.text(cx, 0.19, lbl,
                ha="center", va="center", color=col,
                fontsize=8, fontweight="bold",
                transform=ax.transAxes)


# ── B: Danger Creation (single team) ──────────────────────────────
def _panel_danger(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax,f"Danger Creation — {name}",tc)
    dng = events[
        ((events["is_shot"]==True)|(events["is_key_pass"]==True)) &
        (events["team_id"]==tid) &
        events[["x","y"]].notna().all(axis=1)
    ].copy()
    if dng.empty: return
    shots  = dng[dng["is_shot"]==True]
    kp     = dng[dng["is_key_pass"]==True]
    goals  = shots[shots["is_goal"]==True] if not shots.empty else shots
    if not dng.empty:
        ax.scatter(dng["x"],dng["y"],c=tc,s=120,alpha=0.07,zorder=2,edgecolors="none")
    if not kp.empty:
        ax.scatter(kp["x"],kp["y"],c="#facc15",s=22,marker="^",
                   alpha=0.78,zorder=4,edgecolors="none")
    if not shots.empty:
        ax.scatter(shots["x"],shots["y"],c=tc,s=30,alpha=0.78,
                   zorder=4,edgecolors="white",lw=0.5)
    if not goals.empty:
        ax.scatter(goals["x"],goals["y"],c="#FFD700",s=95,marker="*",
                   zorder=6,edgecolors="white",lw=0.8)
    ax.text(50,-4.5,
            f"Shots:{len(shots)}  Key Pass:{len(kp)}  Goals:{len(goals)}",
            ha="center",va="top",color=tc,fontsize=6.5,fontweight="bold")
    ax.legend(handles=[
        Line2D([0],[0],marker="^",color="w",markerfacecolor="#facc15",
               markersize=6,linestyle="None",label="Key Pass"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=tc,
               markersize=6,linestyle="None",label="Shot"),
        Line2D([0],[0],marker="*",color="w",markerfacecolor="#FFD700",
               markersize=8,linestyle="None",label="Goal"),
    ],fontsize=5.5,facecolor=BG_MID,edgecolor="none",
      labelcolor="white",loc="upper left",markerscale=0.9,framealpha=0.80)


# ── C: Goals & Assists table ───────────────────────────────────────
def _panel_goals_table(ax, events, info):
    ax.set_facecolor(BG_MID); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    _lbl(ax,"Goals & Assists",TEXT_BRIGHT)
    gdf = events[events["is_goal"]==True].copy()
    if gdf.empty:
        ax.text(0.50,0.50,"No goals recorded",ha="center",va="center",
                color=TEXT_DIM,fontsize=10,style="italic",transform=ax.transAxes)
        return
    rows=[]
    for _,r in gdf.sort_values("minute").iterrows():
        is_og  = bool(r.get("is_own_goal",False))
        ben_id = r.get("scoring_team",r["team_id"])
        col    = OG_COLOR if is_og else (C_RED if ben_id==info["home_id"] else C_BLUE)
        gtype  = "🔄OG" if is_og else ("🟡Pen" if r["is_penalty"]
                                        else("🔵Hdr" if r["is_header"] else "⚽"))
        rows.append((r["minute"],r["player"],r["assist_player"],gtype,r["xG"],col,is_og))
    row_h = min(0.78/max(len(rows),1),0.14)
    y = 0.87
    for x_,lbl_,ha_ in [(0.07,"MIN","center"),(0.35,"SCORER","left"),
                         (0.62,"ASSIST","left"),(0.82,"TYPE","center"),(0.95,"xG","right")]:
        ax.text(x_,y,lbl_,ha=ha_,va="center",color=TEXT_DIM,fontsize=7.5,
                fontweight="bold",transform=ax.transAxes)
    y -= 0.025
    ax.plot([0.01,0.99],[y,y],color=GRID_COL,lw=0.8,transform=ax.transAxes)
    y -= 0.008
    for min_,scorer,assist,gtype,xg,col,is_og in rows:
        bg = "#1e0a2e" if is_og else ("#1a0a0a" if col==C_RED else "#060f1e")
        ax.add_patch(plt.Rectangle((0.01,y-row_h*0.85),0.98,row_h*0.82,
            facecolor=bg,edgecolor=col,lw=0.6,alpha=0.9,transform=ax.transAxes))
        cy = y-row_h*0.42
        ax.text(0.07,cy,f"{int(min_)}'",ha="center",va="center",color="white",
                fontsize=8.5,fontweight="bold",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.22",facecolor=col,alpha=0.88,edgecolor="none"))
        ax.text(0.35,cy,_short(str(scorer)) if scorer else "—",
                ha="left",va="center",color=col,fontsize=9,
                fontweight="bold",transform=ax.transAxes)
        ax.text(0.62,cy,_short(str(assist)) if assist else "—",
                ha="left",va="center",color=TEXT_DIM,fontsize=8.5,transform=ax.transAxes)
        ax.text(0.82,cy,gtype,ha="center",va="center",color=col,
                fontsize=9,transform=ax.transAxes)
        ax.text(0.97,cy,f"{xg:.2f}" if xg else "—",ha="right",va="center",
                color="#facc15",fontsize=8.5,transform=ax.transAxes)
        y -= row_h


# ── D: Mini Shot Map (single team) ────────────────────────────────
def _panel_shot_mini(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax,f"Shot Map — {name}",tc)
    shots = events[(events["is_shot"]==True)&(events["team_id"]==tid)&
                   events[["x","y"]].notna().all(axis=1)].copy()
    if shots.empty: return
    for cat,(marker,fc,ec,sz,z,_) in SHOT_STYLE.items():
        sub = shots[shots["shot_category"]==cat]
        if sub.empty: continue
        ax.scatter(sub["x"],sub["y"],c=fc,s=sz*0.15,marker=marker,
                   edgecolors=ec,linewidths=0.8,alpha=0.92,zorder=z)
    n_tot = len(shots); n_sot = int(shots["shot_category"].isin(["Saved","Goal"]).sum())
    xg_t  = round(float(shots["xG"].sum()),2)
    ax.text(50,-4.5,f"Shots:{n_tot}  SoT:{n_sot}  xG:{xg_t}",
            ha="center",va="top",color=tc,fontsize=6.5,fontweight="bold")


# ── E: xG / xGoT tiles (full-width summary) ───────────────────────
def _panel_xg_tiles(ax, events, info, xg_data):
    ax.set_facecolor(BG_DARK); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    hn,an = info["home_name"],info["away_name"]
    hd=xg_data.get(hn,{}); ad=xg_data.get(an,{})
    hs  = events[(events["is_shot"]==True)&(events["team_id"]==info["home_id"])]
    as_ = events[(events["is_shot"]==True)&(events["team_id"]==info["away_id"])]
    hxgot=round(hs[hs["shot_category"].isin(["Saved","Goal"])]["xG"].sum(),2)
    axgot=round(as_[as_["shot_category"].isin(["Saved","Goal"])]["xG"].sum(),2)
    tiles=[
        (hn[:12],hd.get("xG",0),     C_RED,    "xG"),
        (hn[:12],hxgot,              "#1e90ff", "xGoT"),
        (hn[:12],hd.get("on_target",0),C_GREEN, "On Target"),
        (an[:12],axgot,              "#1e90ff", "xGoT"),
        (an[:12],ad.get("xG",0),     C_BLUE,   "xG"),
        (an[:12],ad.get("on_target",0),C_GREEN, "On Target"),
    ]
    tw=1.0/len(tiles)
    for i,(team,val,col,lbl) in enumerate(tiles):
        cx=(i+0.5)*tw
        ax.add_patch(plt.Rectangle((i*tw+0.003,0.04),tw-0.006,0.92,
            facecolor="#0d1117",edgecolor=col,lw=1.2,alpha=0.9,transform=ax.transAxes))
        ax.text(cx,0.72,str(round(val,2) if isinstance(val,float) else val),
                ha="center",va="center",color=col,fontsize=16,fontweight="bold",
                transform=ax.transAxes)
        ax.text(cx,0.40,lbl,ha="center",va="center",color=TEXT_DIM,
                fontsize=9,transform=ax.transAxes)
        ax.text(cx,0.18,team,ha="center",va="center",color="white",
                fontsize=8,transform=ax.transAxes)


# ── F: Pass Map with thirds (single team) ─────────────────────────
def _panel_pass_thirds(ax, events, tid, tc, name):
    _mini_pitch(ax)
    _lbl(ax,f"Pass Map — {name}",tc)
    p = events[
        (events["is_pass"]==True)&(events["team_id"]==tid)&
        events[["x","y","end_x","end_y"]].notna().all(axis=1)
    ].copy()
    if p.empty: return
    zone_cfg=[
        ((0,  33),"#64748b","Own 3rd"),
        ((33, 66),C_GOLD,  "Mid 3rd"),
        ((66,100),C_GREEN, "Final 3rd"),
    ]
    for (x0,x1),col,zlbl in zone_cfg:
        sub=p[(p["x"]>=x0)&(p["x"]<x1)]
        succ=sub[sub["outcome"]=="Successful"]
        fail=sub[sub["outcome"]!="Successful"]
        for df_,alpha,lw in [(succ,0.55,0.9),(fail,0.20,0.5)]:
            for _,r in df_.iterrows():
                ax.annotate("",xy=(r["end_x"],r["end_y"]),xytext=(r["x"],r["y"]),
                    arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,
                                   alpha=alpha,mutation_scale=4),zorder=3)
    ax.axvline(33,color="#475569",lw=0.8,ls=":",alpha=0.6,zorder=5)
    ax.axvline(66,color="#475569",lw=0.8,ls=":",alpha=0.6,zorder=5)
    tot=len(p)
    for (x0,x1),col,zlbl in zone_cfg:
        sub=p[(p["x"]>=x0)&(p["x"]<x1)]
        suc=int((sub["outcome"]=="Successful").sum())
        pct=round(len(sub)/tot*100) if tot else 0
        cx = (x0+x1)/2
        # السطر الأول: اسم المنطقة
        ax.text(cx, -3.5, zlbl,
                ha="center", va="top", color=col,
                fontsize=7, fontweight="bold")
        # السطر الثاني: الإجمالي والنسبة
        ax.text(cx, -8.5, f"{len(sub)} passes ({pct}%)",
                ha="center", va="top", color="#cbd5e1",
                fontsize=6.5)
        # السطر الثالث: الناجحة
        ax.text(cx, -13.0, f"{suc} ✓ completed",
                ha="center", va="top", color=C_GREEN,
                fontsize=6.5)


# ── G: Progressive Passes (single team) ───────────────────────────
def _panel_progressive(ax, events, tid, tc, name):
    _mini_pitch(ax)
    p=events[(events["is_pass"]==True)&(events["team_id"]==tid)&
             (events["outcome"]=="Successful")&
             events[["x","y","end_x","end_y"]].notna().all(axis=1)].copy()
    prog=p[(p["end_x"]-p["x"])>=10].copy() if not p.empty else p
    n=len(prog)
    _lbl(ax,f"{name}: {n} Progressive Passes",tc)
    if prog.empty: return
    def zc(x):
        if x<33: return "#64748b"
        if x<66: return C_GOLD
        return C_GREEN
    for _,r in prog.iterrows():
        ax.annotate("",xy=(r["end_x"],r["end_y"]),xytext=(r["x"],r["y"]),
            arrowprops=dict(arrowstyle="-|>",color=zc(r["x"]),lw=0.9,
                           alpha=0.55,mutation_scale=5),zorder=3)
    n1=int((prog["x"]<33).sum()); n2=int(((prog["x"]>=33)&(prog["x"]<66)).sum())
    n3=int((prog["x"]>=66).sum())
    for i,(lv,cl) in enumerate([
        (f"Own 3rd  {n1}({int(n1/n*100) if n else 0}%)","#64748b"),
        (f"Mid 3rd  {n2}({int(n2/n*100) if n else 0}%)",C_GOLD),
        (f"Final 3rd {n3}({int(n3/n*100) if n else 0}%)",C_GREEN),
    ]):
        ax.text(99,99-i*9,lv,ha="right",va="top",color=cl,fontsize=6.5,
                fontweight="bold",zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",facecolor=BG_DARK,
                          alpha=0.80,edgecolor="none"))


# ── H: Defensive Heatmap (single team) ────────────────────────────
DEFENSIVE_TYPES={
    "Tackle":       (C_RED,    "Tackle"),
    "Interception": (C_BLUE,   "Int."),
    "BallRecovery": ("#22c55e","Recovery"),
    "Clearance":    (C_GOLD,   "Clear."),
    "BlockedShot":  (C_GREEN,  "Block"),
    "Aerial":       ("#a855f7","Aerial"),
    "Challenge":    ("#f97316","Challenge"),
}

def _panel_defensive_heatmap(ax, events, tid, tc, name):
    """
    Defensive Actions Heatmap — single team.
    Each type = unique colour + marker. Legend outside pitch (right side).
    """
    DEF_STYLE = {
        "Tackle":       ("#f43f5e", "D",  "Tackle"),
        "Interception": ("#3b82f6", "^",  "Interception"),
        "BallRecovery": ("#22c55e", "o",  "Ball Recovery"),
        "Clearance":    ("#f59e0b", "s",  "Clearance"),
        "BlockedShot":  ("#a855f7", "P",  "Blocked Shot"),
        "Aerial":       ("#06b6d4", "*",  "Aerial"),
        "Challenge":    ("#fb923c", "v",  "Challenge"),
        "Foul":         ("#94a3b8", "x",  "Foul"),
    }

    # extend xlim to give legend room on the right
    _mini_pitch(ax, bg="#040d04")
    ax.set_xlim(-3, 130)   # override default → extra 27 units for legend
    ax.set_ylim(-10, 113)

    # title — use transAxes so extended xlim doesn't affect it
    ax.text(0.38, 1.025, f"Defensive Actions — {name}",
            ha="center", va="bottom",
            transform=ax.transAxes,
            color=tc, fontsize=8.5, fontweight="bold", zorder=20,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d1117",
                      edgecolor=tc, linewidth=1.0, alpha=0.92))

    def_ev = events[
        (events["team_id"] == tid) &
        (events["type"].isin(DEF_STYLE.keys())) &
        events[["x","y"]].notna().all(axis=1)
    ].copy()

    if def_ev.empty:
        ax.text(50, 50, "No defensive data", ha="center", va="center",
                color=TEXT_DIM, fontsize=8, style="italic"); return

    # ── KDE density background ───────────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap as LSC
    hm_cmap = LSC.from_list("dhm", [
        "#000000","#071020","#0d2a4a","#0f4c2a",
        "#a16207","#dc2626","#ff0a47"], N=256)
    xs, ys = def_ev["x"].values, def_ev["y"].values
    if len(xs) >= 6:
        try:
            from scipy.stats import gaussian_kde
            gx, gy = np.mgrid[0:100:50j, 0:100:50j]
            kde = gaussian_kde(np.vstack([xs, ys]), bw_method=0.22)
            z   = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            ax.contourf(gx, gy, z, levels=10, cmap=hm_cmap, alpha=0.55, zorder=2)
        except Exception:
            pass

    # ── Scatter each type ────────────────────────────────────────
    for dtype, (col, mkr, _lbl) in DEF_STYLE.items():
        sub = def_ev[def_ev["type"] == dtype]
        if sub.empty: continue
        ax.scatter(sub["x"], sub["y"], c=col, marker=mkr, s=40,
                   edgecolors="white", linewidths=0.5, alpha=0.92, zorder=5)

    # ── Legend panel (right of pitch: x 104–128) ─────────────────
    counts = def_ev["type"].value_counts()
    lx0 = 104
    ax.add_patch(plt.Rectangle((lx0, -2), 25, 104,
                                facecolor="#080f08", alpha=0.85,
                                edgecolor="#334155", lw=0.8, zorder=6))
    ax.text(lx0 + 12.5, 99, "LEGEND",
            ha="center", va="top", color=TEXT_DIM,
            fontsize=6.5, fontweight="bold", zorder=7)

    ly = 93
    for dtype, (col, mkr, lbl_txt) in DEF_STYLE.items():
        n = int(counts.get(dtype, 0))
        if n == 0:
            continue
        ax.scatter([lx0 + 3], [ly], c=col, marker=mkr, s=32,
                   edgecolors="white", linewidths=0.5, zorder=8)
        ax.text(lx0 + 6.5, ly, f"{lbl_txt}",
                ha="left", va="center", color=col,
                fontsize=6.2, fontweight="bold", zorder=8)
        ax.text(lx0 + 6.5, ly - 4.5, f"n = {n}",
                ha="left", va="center", color="#94a3b8",
                fontsize=5.5, zorder=8)
        ly -= 13

    # ── Zone footer (below pitch) ─────────────────────────────────
    zone_data = [("Def 3rd", 0, 33), ("Mid 3rd", 33, 66), ("Att 3rd", 66, 100)]
    parts = []
    for zlbl, x0, x1 in zone_data:
        n = int(def_ev[(def_ev["x"] >= x0) & (def_ev["x"] < x1)].shape[0])
        parts.append(f"{zlbl}: {n}")
    ax.text(50, -7, "   ".join(parts),
            ha="center", va="top", color=tc,
            fontsize=6.5, fontweight="bold")


# ── I: Pass Network (single team, mini) ───────────────────────────
def _panel_pass_network(ax, events, tid, tc, name):
    _mini_pitch(ax); _lbl(ax,f"Pass Network — {name}",tc)
    p=events[(events["is_pass"]==True)&(events["team_id"]==tid)&
             (events["outcome"]=="Successful")&
             events[["x","y"]].notna().all(axis=1)].copy()
    if p.empty: return
    avg=p.groupby("player")[["x","y"]].mean()
    cnt=p.groupby("player").size().rename("n")
    avg=avg.join(cnt); mx_n=avg["n"].max() if not avg.empty else 1
    for pl,row in avg.iterrows():
        s=18+row["n"]/mx_n*90
        ax.scatter(row["x"],row["y"],c=tc,s=s,edgecolors="white",lw=0.7,zorder=5)
        nm=pl.split()[-1][:8] if pl else ""
        ax.text(row["x"],row["y"]+3.5,nm,ha="center",va="bottom",
                color="white",fontsize=5.5,zorder=6,
                bbox=dict(boxstyle="round,pad=0.14",facecolor="#000",
                          alpha=0.68,edgecolor="none"))


# ── J: Avg Positions (single team) ────────────────────────────────
def _panel_avg_position(ax, events, tid, tc, name):
    _mini_pitch(ax); _lbl(ax,f"Avg Positions — {name}",tc)
    ev=events[(events["team_id"]==tid)&events[["x","y"]].notna().all(axis=1)].copy()
    if ev.empty: return
    avg=ev.groupby("player")[["x","y"]].mean()
    cnt=ev.groupby("player").size(); mx=cnt.max() if not cnt.empty else 1
    for pl,row in avg.iterrows():
        n=cnt.get(pl,1)
        ax.scatter(row["x"],row["y"],c=tc,s=28+n/mx*82,
                   edgecolors="white",lw=0.7,alpha=0.88,zorder=4)
        nm=pl.split()[-1][:7] if pl else ""
        ax.text(row["x"],row["y"]+3.2,nm,ha="center",va="bottom",
                color="white",fontsize=5.5,zorder=5,
                bbox=dict(boxstyle="round,pad=0.12",facecolor="#000",
                          alpha=0.68,edgecolor="none"))


# ── K: Match Statistics table ──────────────────────────────────────
def _panel_match_stats(ax, events, info, xg_data):
    ax.set_facecolor(BG_DARK); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    hn,an = info["home_name"],info["away_name"]
    hid,aid = info["home_id"],info["away_id"]
    hd=xg_data.get(hn,{}); ad=xg_data.get(an,{})
    hp  = int(events[(events["is_pass"]==True)&(events["team_id"]==hid)].shape[0])
    ap  = int(events[(events["is_pass"]==True)&(events["team_id"]==aid)].shape[0])
    tot_p=(hp+ap) or 1
    hps = int(events[(events["is_pass"]==True)&(events["team_id"]==hid)&
                      (events["outcome"]=="Successful")].shape[0])
    aps = int(events[(events["is_pass"]==True)&(events["team_id"]==aid)&
                      (events["outcome"]=="Successful")].shape[0])
    hkp = int(events[(events["is_key_pass"]==True)&(events["team_id"]==hid)].shape[0])
    akp = int(events[(events["is_key_pass"]==True)&(events["team_id"]==aid)].shape[0])
    hxt = round(events[(events["team_id"]==hid) & events["xT"].notna()]["xT"].sum(), 2) \
          if "xT" in events.columns else 0
    axt = round(events[(events["team_id"]==aid)&events["xT"].notna()]["xT"].sum(),2) \
          if "xT" in events.columns else 0
    stats=[
        ("Possession",  f"{round(hp/tot_p*100,1)}%",  f"{round(ap/tot_p*100,1)}%",
         hp/tot_p,      C_GOLD),
        ("Passes (Acc)",f"{hp}({hps})",                f"{ap}({aps})",
         hp/((hp+ap) or 1), "#94a3b8"),
        ("Shots (SoT)", f"{hd.get('shots',0)}({hd.get('on_target',0)})",
                        f"{ad.get('shots',0)}({ad.get('on_target',0)})",
         hd.get("shots",0)/((hd.get("shots",0)+ad.get("shots",0)) or 1),"#64748b"),
        ("xG",          str(hd.get("xG",0)),           str(ad.get("xG",0)),
         hd.get("xG",0)/((hd.get("xG",0)+ad.get("xG",0)) or 1),"#a855f7"),
        ("xT",          str(hxt),                       str(axt),
         hxt/((hxt+axt) or 1),"#22c55e"),
        ("Key Passes",  str(hkp),                       str(akp),
         hkp/((hkp+akp) or 1),"#1e90ff"),
        ("Big Chances", str(hd.get("big_chances",0)),   str(ad.get("big_chances",0)),
         hd.get("big_chances",0)/((hd.get("big_chances",0)+ad.get("big_chances",0)) or 1),
         "#f43f5e"),
    ]
    ax.text(0.50,0.99,"MATCH STATISTICS",ha="center",va="top",
            color=TEXT_BRIGHT,fontsize=9.5,fontweight="bold",transform=ax.transAxes)
    ax.text(0.10,0.94,hn[:13],ha="left",va="top",color=C_RED,fontsize=8,
            fontweight="bold",transform=ax.transAxes)
    ax.text(0.90,0.94,an[:13],ha="right",va="top",color=C_BLUE,fontsize=8,
            fontweight="bold",transform=ax.transAxes)
    ax.plot([0.02,0.98],[0.90,0.90],color=GRID_COL,lw=0.8,transform=ax.transAxes)
    y=0.85; step=0.118
    for lbl_,hv,av,hr,col in stats:
        # Full-width strip background
        strip_x, strip_y, strip_h = 0.02, y - 0.050, 0.052
        strip_w = 0.96
        # Home side fill
        ax.add_patch(plt.Rectangle(
            (strip_x, strip_y), strip_w * hr, strip_h,
            facecolor=C_RED, alpha=0.32, transform=ax.transAxes))
        # Away side fill
        ax.add_patch(plt.Rectangle(
            (strip_x + strip_w * hr, strip_y), strip_w * (1 - hr), strip_h,
            facecolor=C_BLUE, alpha=0.32, transform=ax.transAxes))
        # Thin outline around full strip
        ax.add_patch(plt.Rectangle(
            (strip_x, strip_y), strip_w, strip_h,
            facecolor="none", edgecolor=col, lw=0.7,
            alpha=0.55, transform=ax.transAxes))
        # Stat label — centred INSIDE the strip
        ax.text(0.50, y - 0.024, lbl_,
                ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold",
                transform=ax.transAxes,
                zorder=5,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#000000")])
        # Home value — left edge inside strip
        ax.text(0.045, y - 0.024, str(hv),
                ha="left", va="center", color="#ffd6d6",
                fontsize=9, fontweight="bold", transform=ax.transAxes, zorder=5)
        # Away value — right edge inside strip
        ax.text(0.955, y - 0.024, str(av),
                ha="right", va="center", color="#d6e8ff",
                fontsize=9, fontweight="bold", transform=ax.transAxes, zorder=5)
        y -= step


# ── L: xT per Minute (both teams, full-width) ─────────────────────
def _panel_xt_minute(ax, events, info):
    """
    Diverging bar chart: Home xT bars up (red), Away xT bars down (blue).
    Team names + totals shown in clear top-left / top-right boxes.
    HT and FT markers with labels above the bars.
    """
    hn, an   = info["home_name"], info["away_name"]
    hid, aid = info["home_id"],   info["away_id"]

    ax.set_facecolor(BG_DARK)
    _lbl(ax, "xT per Minute  (▲ Home  |  ▼ Away)", TEXT_BRIGHT)

    if "xT" not in events.columns:
        ax.text(0.5, 0.5, "No xT data", ha="center", va="center",
                color=TEXT_DIM, transform=ax.transAxes, fontsize=10); return

    xt = events[
        events["xT"].notna() & (events["xT"] > 0) &
        (events["outcome"] == "Successful")
    ].copy()

    hxt = xt[xt["team_id"]==hid].groupby("minute")["xT"].sum()
    axt = xt[xt["team_id"]==aid].groupby("minute")["xT"].sum()
    mins = list(range(1, 96))
    h_vals = [hxt.get(m, 0) for m in mins]
    a_vals = [-axt.get(m, 0) for m in mins]

    ax.bar(mins, h_vals, color=C_RED,  alpha=0.72, width=0.85, zorder=3)
    ax.bar(mins, a_vals, color=C_BLUE, alpha=0.72, width=0.85, zorder=3)
    ax.axhline(0, color="#94a3b8", lw=0.9, alpha=0.55, zorder=4)
    # 5-minute rolling average overlay
    import pandas as _pd2
    _hv = _pd2.Series(h_vals).rolling(5, center=True, min_periods=1).mean()
    _av = _pd2.Series(a_vals).rolling(5, center=True, min_periods=1).mean()
    ax.plot(mins, _hv, color=C_RED,  lw=2.0, alpha=0.92, zorder=5)
    ax.plot(mins, _av, color=C_BLUE, lw=2.0, alpha=0.92, zorder=5)

    # HT / FT markers
    ymax = max(max(h_vals + [0.001]), abs(min(a_vals + [-0.001])))
    for xp, lb in [(45, "HT"), (90, "FT")]:
        ax.axvline(xp, color=C_GOLD, lw=1.2, ls="--", alpha=0.65, zorder=2)
        ax.text(xp + 0.8, ymax * 0.90, lb,
                color=C_GOLD, fontsize=7.5, fontweight="bold", va="top")

    # xT totals
    ht_total = round(xt[xt["team_id"]==hid]["xT"].sum(), 3)
    at_total = round(xt[xt["team_id"]==aid]["xT"].sum(), 3)

    for xpos, name, col, ha_, xT_val in [
        (0.01, hn, C_RED,  "left",  ht_total),
        (0.99, an, C_BLUE, "right", at_total),
    ]:
        ax.text(xpos, 0.97, f"{name[:12]}",
                transform=ax.transAxes, ha=ha_, va="top",
                color=col, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_MID,
                          alpha=0.90, edgecolor=col, lw=0.8))
        ax.text(xpos, 0.80, f"xT: {xT_val}",
                transform=ax.transAxes, ha=ha_, va="top",
                color=TEXT_DIM, fontsize=8)

    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=8, labelpad=4)
    ax.set_ylabel("xT / min", color=TEXT_DIM, fontsize=8, labelpad=4)
    ax.set_xlim(0, 96)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color(GRID_COL)
    ax.grid(axis="y", alpha=0.10, color=GRID_COL)


# ── M: Crosses (both teams) ────────────────────────────────────────
def _panel_crosses(ax, events, info):
    _mini_pitch(ax); _lbl(ax,"Crosses",TEXT_DIM)
    if "is_cross" not in events.columns:
        ax.text(50,50,"No cross data",ha="center",va="center",
                color=TEXT_DIM,fontsize=8,style="italic"); return
    crs=events[(events["is_cross"]==True)&
               events[["x","y","end_x","end_y"]].notna().all(axis=1)].copy()
    if crs.empty:
        ax.text(50,50,"No crosses recorded",ha="center",va="center",
                color=TEXT_DIM,fontsize=8,style="italic"); return
    for tid,col in [(info["home_id"],C_RED),(info["away_id"],C_BLUE)]:
        tc_=crs[crs["team_id"]==tid]
        for _,r in tc_.iterrows():
            succ=r.get("outcome","")=="Successful"
            ax.annotate("",xy=(r["end_x"],r["end_y"]),xytext=(r["x"],r["y"]),
                arrowprops=dict(arrowstyle="-|>",color=col,
                                lw=1.3 if succ else 0.6,
                                alpha=0.82 if succ else 0.30,mutation_scale=6),zorder=4)
        eff=int(tc_[tc_["outcome"]=="Successful"].shape[0])
        nm=info["home_name"][:8] if tid==info["home_id"] else info["away_name"][:8]
        xp=2 if tid==info["home_id"] else 98
        ax.text(xp,-4.5,f"{nm}: {len(tc_)} ({eff} eff.)",
                ha="left" if tid==info["home_id"] else "right",
                va="top",color=col,fontsize=6.5,fontweight="bold")




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
        ax.text(50, 50, "No cross data", ha="center", va="center",
                color=TEXT_DIM, fontsize=8, style="italic"); return

    crs = events[
        (events["is_cross"] == True) &
        (events["team_id"] == tid) &
        events[["x","y","end_x","end_y"]].notna().all(axis=1)
    ].copy()

    if crs.empty:
        ax.text(50, 50, "No crosses recorded", ha="center", va="center",
                color=TEXT_DIM, fontsize=8, style="italic"); return

    succ_crs = crs[crs["outcome"] == "Successful"]
    fail_crs = crs[crs["outcome"] != "Successful"]
    n_succ   = len(succ_crs)
    n_fail   = len(fail_crs)
    n_total  = len(crs)

    # ── Draw arrows ─────────────────────────────────────────────
    for _,r in fail_crs.iterrows():
        ax.annotate("", xy=(r["end_x"], r["end_y"]), xytext=(r["x"], r["y"]),
            arrowprops=dict(arrowstyle="-|>", color=tc,
                            lw=0.6, alpha=0.28, mutation_scale=5), zorder=3)

    for _,r in succ_crs.iterrows():
        # Glow trail
        ax.annotate("", xy=(r["end_x"], r["end_y"]), xytext=(r["x"], r["y"]),
            arrowprops=dict(arrowstyle="-|>", color="white",
                            lw=4.0, alpha=0.08, mutation_scale=10), zorder=4)
        ax.annotate("", xy=(r["end_x"], r["end_y"]), xytext=(r["x"], r["y"]),
            arrowprops=dict(arrowstyle="-|>", color=tc,
                            lw=1.6, alpha=0.88, mutation_scale=8), zorder=5)

    # ── Origin dots ─────────────────────────────────────────────
    ax.scatter(crs["x"], crs["y"], c=tc, s=22, alpha=0.72,
               edgecolors="white", lw=0.5, zorder=6)

    # ── Left / Right breakdown ───────────────────────────────────
    left_n  = int((crs["y"] < 40).sum())
    right_n = int((crs["y"] > 60).sum())
    centre_n = n_total - left_n - right_n

    for xp, yp, ha_, txt, col_ in [
        (2,  98, "left",  f"Left: {left_n}",    tc),
        (50, 98, "center",f"Ctr: {centre_n}",   TEXT_DIM),
        (98, 98, "right", f"Right: {right_n}",  tc),
    ]:
        ax.text(xp, yp, txt, ha=ha_, va="top", color=col_,
                fontsize=6.8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.20", facecolor=BG_DARK,
                          alpha=0.78, edgecolor="none"), zorder=7)

    # ── Summary footer ───────────────────────────────────────────
    acc_pct = round(n_succ / n_total * 100) if n_total else 0
    ax.text(50, -5.0,
            f"Total: {n_total}  ●  Effective: {n_succ} ({acc_pct}%)  ●  Missed: {n_fail}",
            ha="center", va="top", color=tc,
            fontsize=7, fontweight="bold")

    # ── Legend ───────────────────────────────────────────────────
    ax.legend(handles=[
        Line2D([0],[0], color=tc,      lw=1.8, alpha=0.92, label=f"Effective ({n_succ})"),
        Line2D([0],[0], color=tc,      lw=0.6, alpha=0.30, label=f"Missed ({n_fail})"),
    ], fontsize=6.5, facecolor=BG_MID, edgecolor="none",
       labelcolor="white", loc="upper right", framealpha=0.85, markerscale=0.9)

# ── N: Territorial Control bar ────────────────────────────────────
def _panel_territorial(ax, events, info):
    """
    Horizontal stacked bar per zone.
    Home (red) = left portion; Away (blue) = right portion.
    Team names at top; percentages and counts labelled on bars.
    """
    ax.set_facecolor(BG_MID)
    hn, an   = info["home_name"], info["away_name"]
    hid, aid = info["home_id"],   info["away_id"]

    # ── Team name headers ─────────────────────────────────────────
    ax.text(0.20, 1.08, hn[:14], ha="center", va="bottom",
            transform=ax.transAxes, color=C_RED,
            fontsize=9, fontweight="bold")
    ax.text(0.80, 1.08, an[:14], ha="center", va="bottom",
            transform=ax.transAxes, color=C_BLUE,
            fontsize=9, fontweight="bold")
    ax.text(0.50, 1.08, "Territorial Control", ha="center", va="bottom",
            transform=ax.transAxes, color=TEXT_DIM,
            fontsize=8, fontstyle="italic")

    zones = [
        ("Own Third",   (0,  33)),
        ("Mid Third",   (33, 66)),
        ("Final Third", (66, 100)),
    ]
    ev = events[events["x"].notna()].copy()

    for i, (zlbl, (x0, x1)) in enumerate(zones):
        hev = ev[(ev["team_id"]==hid) & (ev["x"]>=x0) & (ev["x"]<x1)]
        aev = ev[(ev["team_id"]==aid) & (ev["x"]>=x0) & (ev["x"]<x1)]
        h_n = len(hev); a_n = len(aev)
        tot = (h_n + a_n) or 1
        hr  = h_n / tot

        # bars — with slight glow edge
        ax.barh(i, hr,   height=0.64, color=C_RED,    alpha=0.85, left=0,
                edgecolor="#ff6b7a", linewidth=0.6)
        ax.barh(i, 1-hr, height=0.64, color=C_BLUE,   alpha=0.85, left=hr,
                edgecolor="#5ba3ff", linewidth=0.6)

        # % inside bar
        if hr > 0.08:
            ax.text(hr/2, i, f"{h_n}  ({hr*100:.0f}%)",
                    ha="center", va="center", color="white",
                    fontsize=8.5, fontweight="bold")
        if (1-hr) > 0.08:
            ax.text(hr + (1-hr)/2, i, f"({(1-hr)*100:.0f}%)  {a_n}",
                    ha="center", va="center", color="white",
                    fontsize=8.5, fontweight="bold")

    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels([z[0] for z in zones],
                       color=TEXT_BRIGHT, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.tick_params(left=False)
    ax.set_ylim(-0.5, len(zones) - 0.4)
    for sp in ["top","right","bottom","left"]:
        ax.spines[sp].set_visible(False)
    ax.set_facecolor(BG_MID)


# ── O: Possession Donut (single team) ─────────────────────────────
def _panel_possession_donut(ax, events, tid, tc, name):
    ax.set_facecolor(BG_MID)
    _lbl(ax,f"Ball Touches — {name}",tc)
    ev=events[events[["x","y"]].notna().all(axis=1)].copy()
    total=len(ev); team=int((ev["team_id"]==tid).sum())
    pct=round(team/total*100,1) if total else 0
    ax.pie([pct,100-pct],colors=[tc,"#1e2836"],startangle=90,
           wedgeprops=dict(width=0.42,edgecolor=BG_DARK,lw=1.5))
    ax.text(0,0,f"{pct}%",ha="center",va="center",
            color="white",fontsize=14,fontweight="bold")
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)


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
    ax.text(50, 50, "Goalkeeper Saves",
            ha="center", va="top", color=TEXT_DIM,
            fontsize=9, fontweight="bold")

    GW, GH = 36, 22   # goal width / height in axes units

    # (frame_cx, frame_by, gk_team_id, shoot_team_id, colour, gk_label)
    FRAMES = [
        (26, 10, info["home_id"], info["away_id"], C_RED,  info["home_name"]),
        (74, 10, info["away_id"], info["home_id"], C_BLUE, info["away_name"]),
    ]

    SHOT_MKR = {
        "VolleyShot":   ("D", 50, "Volley"),
        "Header":       ("^", 50, "Header"),
        "FreekickShot": ("p", 50, "Free Kick"),
        "PenaltyShot":  ("*", 70, "Penalty"),
    }
    DEFAULT_MKR = ("o", 44, "Open Play")

    xg_cmap = mcolors.LinearSegmentedColormap.from_list(
        "xg_sv", ["#22c55e", "#facc15", "#ef4444"], N=256)

    for cx, by, gk_tid, shoot_tid, col, gk_name in FRAMES:
        gx0, gx1 = cx - GW/2, cx + GW/2
        gy0, gy1 = by, by + GH

        # ── Goal frame ───────────────────────────────────────────
        ax.plot([gx0, gx1, gx1, gx0, gx0],
                [gy0, gy0, gy1, gy1, gy0],
                color="white", lw=2.8, alpha=0.95, zorder=4,
                solid_capstyle="round")

        # Net grid
        for nx in np.linspace(gx0 + 2.5, gx1 - 2.5, 7):
            ax.plot([nx, nx], [gy0, gy1],
                    color="#374151", lw=0.55, alpha=0.50, zorder=2)
        for ny in np.linspace(gy0 + 2.5, gy1 - 2.5, 3):
            ax.plot([gx0, gx1], [ny, ny],
                    color="#374151", lw=0.55, alpha=0.50, zorder=2)

        # Zone dividers: thirds (vertical) + half-height (horizontal)
        for xd in [gx0 + GW/3, gx0 + 2*GW/3]:
            ax.plot([xd, xd], [gy0, gy1],
                    color="#6b7280", lw=0.9, ls="--", alpha=0.60, zorder=3)
        ymid = by + GH / 2
        ax.plot([gx0, gx1], [ymid, ymid],
                color="#6b7280", lw=0.9, ls="--", alpha=0.60, zorder=3)

        # Zone text labels
        for xi, xt in [(gx0 + GW*0.17, "L"), (cx, "C"), (gx0 + GW*0.83, "R")]:
            ax.text(xi, gy0 + 1.5, xt, ha="center", va="bottom",
                    color="#9ca3af", fontsize=6, zorder=5)
        ax.text(gx0 - 1.0, ymid + GH*0.25, "High",
                ha="right", va="center", color="#9ca3af",
                fontsize=5.5, rotation=90, zorder=5)
        ax.text(gx0 - 1.0, ymid - GH*0.25, "Low",
                ha="right", va="center", color="#9ca3af",
                fontsize=5.5, rotation=90, zorder=5)

        # ── All saves by shoot_tid ────────────────────────────────
        saves_all = events[
            (events["team_id"] == shoot_tid) &
            (events["shot_category"] == "Saved")
        ].copy()
        n_saves = len(saves_all)

        # Team label + save count below frame
        ax.text(cx, gy0 - 2.5, f"{gk_name[:14]}",
                ha="center", va="top", color=col,
                fontsize=8, fontweight="bold")
        ax.text(cx, gy0 - 5.5, f"{n_saves} saves",
                ha="center", va="top", color=TEXT_DIM, fontsize=7)

        if saves_all.empty:
            ax.text(cx, by + GH/2, "No saves",
                    ha="center", va="center", color=TEXT_DIM, fontsize=7)
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

            ax.scatter([sx_i], [sy_i], c=[rgba], marker=mkr, s=sz,
                       edgecolors="white", linewidths=0.8,
                       alpha=0.95, zorder=6)
            ax.text(sx_i, sy_i + 1.5, f"{xg_v:.2f}",
                    ha="center", va="bottom", color="#fde68a",
                    fontsize=4.5, fontweight="bold", zorder=7)

    # ── Shared legend at bottom ───────────────────────────────────
    # xG colour scale
    ax.text(50, -2, "Colour = xG intensity:", ha="center", va="top",
            color=TEXT_DIM, fontsize=6, style="italic")
    for i, (xg_v, lbl_) in enumerate([(0.05,"Low"), (0.30,"Mid"), (0.65,"High")]):
        xi = 30 + i * 14
        rgba = xg_cmap(xg_v)
        ax.scatter([xi], [-5.5], c=[rgba], marker="o", s=30,
                   edgecolors="white", linewidths=0.4, zorder=5)
        ax.text(xi + 1.8, -5.5, lbl_, ha="left", va="center",
                color=TEXT_DIM, fontsize=5.5)

    ax.text(50, -7.5, "Shape = Shot type:", ha="center", va="top",
            color=TEXT_DIM, fontsize=6, style="italic")
    shape_items = [("o","Open Play"),("^","Header"),
                   ("D","Volley"),("p","Free Kick"),("*","Penalty")]
    for i, (mkr, lbl_) in enumerate(shape_items):
        xi = 12 + i * 17
        ax.scatter([xi], [-10.5], c="white", marker=mkr, s=26,
                   edgecolors="#6b7280", linewidths=0.5, zorder=5)
        ax.text(xi + 2, -10.5, lbl_, ha="left", va="center",
                color=TEXT_DIM, fontsize=5.5)


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
    fig.clear(); fig.patch.set_facecolor(BG_DARK)
    hn, an  = info["home_name"], info["away_name"]
    hid,aid = info["home_id"],   info["away_id"]
    hg  = xg_data.get(hn, {}).get("goals", 0)
    ag  = xg_data.get(an, {}).get("goals", 0)
    hxg = xg_data.get(hn, {}).get("xG",   0.0)
    axg = xg_data.get(an, {}).get("xG",   0.0)

    _page_header(fig, hn, an, hg, ag, hxg, axg,
                 info["home_form"], info["away_form"],
                 info["venue"], status, 1, "⚽ ATTACK REPORT")

    gs = GridSpec(4, 3, figure=fig,
                  left=0.04, right=0.97,
                  top=0.945, bottom=0.030,
                  hspace=0.65, wspace=0.13,
                  height_ratios=[0.70, 1.15, 1.00, 0.55])

    # ── Row 0: Shot Comparison tiles (full width) ──────────────────
    # ✅ UNIQUE — Fig 4 uses bar chart; these are per-metric tiles
    _panel_shot_comparison(fig.add_subplot(gs[0, :]), events, info, xg_data)

    # ── Row 1: Danger Home | GK Saves | Danger Away ────────────────
    # ✅ UNIQUE — none of these exist in Figs 1-10
    _panel_danger(   fig.add_subplot(gs[1, 0]), events, hid, C_RED,  hn)
    _panel_gk_saves( fig.add_subplot(gs[1, 1]), events, info)
    _panel_danger(   fig.add_subplot(gs[1, 2]), events, aid, C_BLUE, an)

    # ── Row 2: Zone 14 Home | Avg Position Home+Away | Zone 14 Away ─
    # ✅ UNIQUE — not in 1-10
    _panel_zone14(        fig.add_subplot(gs[2, 0]), events, hid, C_RED,  hn)
    _panel_avg_pos_dual(  fig.add_subplot(gs[2, 1]), events, info)          # both teams
    _panel_zone14(        fig.add_subplot(gs[2, 2]), events, aid, C_BLUE,  an)

    # ── Row 3: xG / xGoT / OnTarget tiles (full width) ─────────────
    # ✅ UNIQUE FORMAT — Fig 4 has bar chart; tiles show absolute values per metric
    _panel_xg_tiles( fig.add_subplot(gs[3, :]), events, info, xg_data)

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
    fig.clear(); fig.patch.set_facecolor(BG_DARK)
    hn, an  = info["home_name"], info["away_name"]
    hid,aid = info["home_id"],   info["away_id"]
    hg  = xg_data.get(hn, {}).get("goals", 0)
    ag  = xg_data.get(an, {}).get("goals", 0)
    hxg = xg_data.get(hn, {}).get("xG",   0.0)
    axg = xg_data.get(an, {}).get("xG",   0.0)

    _page_header(fig, hn, an, hg, ag, hxg, axg,
                 info["home_form"], info["away_form"],
                 info["venue"], status, 2, "🔵 POSSESSION & DEFENSE REPORT")

    gs = GridSpec(5, 3, figure=fig,
                  left=0.04, right=0.97,
                  top=0.945, bottom=0.030,
                  hspace=0.60, wspace=0.13,
                  height_ratios=[0.80, 1.0, 1.0, 1.0, 1.0])

    # ── Row 0: Match Stats | Territorial Control | Possession Donuts ──
    # ✅ All UNIQUE — not in 1-10
    _panel_match_stats(    fig.add_subplot(gs[0, 0]), events, info, xg_data)
    _panel_territorial(    fig.add_subplot(gs[0, 1]), events, info)
    _panel_donut_dual(     fig.add_subplot(gs[0, 2]), events, info)

    # ── Row 1: Pass Map/Thirds Home | xT per Minute | Pass Map/Thirds Away
    # ✅ Pass thirds = NEW (Figs 5-6 have no third breakdown)
    # ✅ xT/min = UNIQUE (Fig 1 = cumulative xG, not xT per minute)
    _panel_pass_thirds(fig.add_subplot(gs[1, 0]), events, hid, C_RED,  hn)
    _panel_xt_minute(  fig.add_subplot(gs[1, 1]), events, info)
    _panel_pass_thirds(fig.add_subplot(gs[1, 2]), events, aid, C_BLUE, an)

    # ── Row 2: Progressive Home | Crosses (split) | Progressive Away ──
    # ✅ All UNIQUE — crosses now shown per team in side panels
    _panel_crosses_team(fig.add_subplot(gs[2, 0]), events, hid, C_RED,  hn)
    _panel_progressive( fig.add_subplot(gs[2, 1]), events, hid, C_RED,  hn)
    _panel_crosses_team(fig.add_subplot(gs[2, 2]), events, aid, C_BLUE, an)

    # ── Row 3: Defensive HM Home | (legend) | Defensive HM Away ──────
    # ✅ UNIQUE
    _panel_defensive_heatmap(fig.add_subplot(gs[3, 0]), events, hid, C_RED,  hn)
    _panel_def_legend(       fig.add_subplot(gs[3, 1]))
    _panel_defensive_heatmap(fig.add_subplot(gs[3, 2]), events, aid, C_BLUE, an)

    # ── Row 4: Avg Position Home | (separator) | Avg Position Away ────
    # ✅ UNIQUE
    _panel_avg_position(fig.add_subplot(gs[4, 0]), events, hid, C_RED,  hn)
    _panel_avg_position(fig.add_subplot(gs[4, 2]), events, aid, C_BLUE, an)
    # centre: defensive action counts table
    _panel_def_counts(  fig.add_subplot(gs[4, 1]), events, info)

    _watermark(fig)


# ═══════════════════════════════════════════════════════════════════
#  ADDITIONAL PANEL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _panel_zone14(ax, events, tid, tc, name):
    """Zone 14 + Half-Spaces — single team."""
    from matplotlib.patches import Rectangle as Rect
    _mini_pitch(ax)
    _lbl(ax, f"Zone 14 & Half-Spaces — {name}", tc)
    ax.add_patch(Rect((66,33),17,34, facecolor=tc,      alpha=0.22,
                       edgecolor=tc,      lw=1.5, zorder=2))
    ax.add_patch(Rect((66,67),17,13, facecolor="#a855f7",alpha=0.22,
                       edgecolor="#a855f7",lw=1.2, zorder=2))
    ax.add_patch(Rect((66,20),17,13, facecolor="#a855f7",alpha=0.22,
                       edgecolor="#a855f7",lw=1.2, zorder=2))
    ev = events[(events["team_id"]==tid) &
                events[["x","y"]].notna().all(axis=1)].copy()
    if ev.empty: return
    z14 = ev[ev["x"].between(66,83) & ev["y"].between(33,67)]
    lhs = ev[ev["x"].between(66,83) & ev["y"].between(67,80)]
    rhs = ev[ev["x"].between(66,83) & ev["y"].between(20,33)]
    if not z14.empty:
        ax.scatter(z14["x"], z14["y"], c=tc, s=10, alpha=0.55,
                   zorder=4, edgecolors="none")
    for val, yx, yy, col in [
        (len(z14), 74.5, 50,    tc),
        (len(lhs), 74.5, 73.5, "#a855f7"),
        (len(rhs), 74.5, 26.5, "#a855f7"),
    ]:
        ax.text(yx, yy, str(val), ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.35", facecolor=col,
                          alpha=0.88, edgecolor="white", lw=0.9))
    ax.text(50, -5.5,
            f"Zone14:{len(z14)}  L.HalfSpace:{len(lhs)}  R.HalfSpace:{len(rhs)}",
            ha="center", va="top", color=tc, fontsize=6.5, fontweight="bold")


def _panel_avg_pos_dual(ax, events, info):
    """Average positions — both teams on same mini-pitch (overview)."""
    _mini_pitch(ax)
    _lbl(ax, "Avg Positions (Both Teams)", TEXT_DIM)
    for tid, tc in [(info["home_id"], C_RED), (info["away_id"], C_BLUE)]:
        ev = events[(events["team_id"]==tid) &
                    events[["x","y"]].notna().all(axis=1)].copy()
        if ev.empty: continue
        avg = ev.groupby("player")[["x","y"]].mean()
        cnt = ev.groupby("player").size(); mx = cnt.max() if not cnt.empty else 1
        for pl, row in avg.iterrows():
            n = cnt.get(pl, 1)
            ax.scatter(row["x"], row["y"], c=tc,
                       s=20 + n/mx*55, edgecolors="white",
                       lw=0.5, alpha=0.88, zorder=4)


def _panel_donut_dual(ax, events, info):
    """Two possession donuts side-by-side inside one axes."""
    ax.set_facecolor(BG_MID); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ev    = events[events[["x","y"]].notna().all(axis=1)].copy()
    total = len(ev)
    for i, (tid, tc, name) in enumerate([
        (info["home_id"], C_RED,  info["home_name"]),
        (info["away_id"], C_BLUE, info["away_name"]),
    ]):
        pct = round(int((ev["team_id"]==tid).sum()) / total * 100, 1) if total else 0
        # sub-axes inside this cell
        cx  = 0.25 + i * 0.50
        sub = ax.inset_axes([cx - 0.20, 0.12, 0.40, 0.72])
        sub.pie([pct, 100-pct], colors=[tc, "#1e2836"], startangle=90,
                wedgeprops=dict(width=0.40, edgecolor=BG_DARK, lw=1.5))
        sub.text(0, 0, f"{pct}%", ha="center", va="center",
                 color="white", fontsize=12, fontweight="bold")
        sub.set_xlim(-1.3, 1.3); sub.set_ylim(-1.3, 1.3)
        ax.text(cx, 0.07, name[:12], ha="center", va="top",
                color=tc, fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.text(cx, 0.94, "Ball Touches", ha="center", va="top",
                color=TEXT_DIM, fontsize=7.5, transform=ax.transAxes)


def _panel_def_legend(ax):
    """Legend for defensive heatmap types."""
    ax.set_facecolor(BG_MID); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.50, 0.97, "Defensive Actions", ha="center", va="top",
            color=TEXT_BRIGHT, fontsize=9, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.50, 0.88, "Legend", ha="center", va="top",
            color=TEXT_DIM, fontsize=8, transform=ax.transAxes)
    items = list(DEFENSIVE_TYPES.items())
    y = 0.78
    for dtype, (col, short) in items:
        ax.add_patch(plt.Circle((0.12, y), 0.04,
                                facecolor=col, transform=ax.transAxes,
                                zorder=3))
        ax.text(0.20, y, dtype, ha="left", va="center", color=col,
                fontsize=8, fontweight="bold", transform=ax.transAxes)
        y -= 0.11
    ax.text(0.50, 0.08,
            "Hot = High density\nCool = Low density",
            ha="center", va="bottom", color=TEXT_DIM, fontsize=7.5,
            transform=ax.transAxes, style="italic")


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

    hn, an   = info["home_name"], info["away_name"]
    hid, aid = info["home_id"],   info["away_id"]

    n_rows = len(DEFENSIVE_TYPES)
    # Layout: headers at 0.88, separator at 0.84, rows from 0.80 down
    ROW_START = 0.80
    ROW_STEP  = 0.80 / (n_rows + 0.5)

    # ── Column headers ────────────────────────────────────────────
    ax.text(0.22, 0.91, hn[:13], ha="center", va="center",
            transform=ax.transAxes, color=C_RED,
            fontsize=9, fontweight="bold")
    ax.text(0.78, 0.91, an[:13], ha="center", va="center",
            transform=ax.transAxes, color=C_BLUE,
            fontsize=9, fontweight="bold")
    ax.text(0.50, 0.91, "Action", ha="center", va="center",
            transform=ax.transAxes, color=TEXT_DIM,
            fontsize=8)

    # separator
    ax.plot([0.02, 0.98], [0.87, 0.87],
            color=GRID_COL, lw=1.0, transform=ax.transAxes)

    y = ROW_START
    for dtype, (col, short) in DEFENSIVE_TYPES.items():
        h_n = int(events[(events["team_id"]==hid) &
                         (events["type"]==dtype)].shape[0])               if "type" in events.columns else 0
        a_n = int(events[(events["team_id"]==aid) &
                         (events["type"]==dtype)].shape[0])
        tot = (h_n + a_n) or 1
        hr  = h_n / tot
        bh  = ROW_STEP * 0.50   # bar height in axes fraction

        # Home bar (red, left side of centre)
        ax.add_patch(plt.Rectangle(
            (0.34, y - bh/2), (hr * 0.32), bh,
            facecolor=C_RED, alpha=0.65,
            transform=ax.transAxes, zorder=2))
        # Away bar (blue, right side of centre)
        ax.add_patch(plt.Rectangle(
            (0.34 + hr*0.32, y - bh/2), ((1-hr) * 0.32), bh,
            facecolor=C_BLUE, alpha=0.65,
            transform=ax.transAxes, zorder=2))

        # Home count (left)
        ax.text(0.22, y, str(h_n),
                ha="center", va="center",
                transform=ax.transAxes, color=C_RED,
                fontsize=10, fontweight="bold", zorder=4)
        # Action name (centre)
        ax.text(0.50, y, short,
                ha="center", va="center",
                transform=ax.transAxes, color=col,
                fontsize=8, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.20",
                          facecolor=BG_DARK, alpha=0.85,
                          edgecolor=col, lw=0.8))
        # Away count (right)
        ax.text(0.78, y, str(a_n),
                ha="center", va="center",
                transform=ax.transAxes, color=C_BLUE,
                fontsize=10, fontweight="bold", zorder=4)

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
    # subtle box fills
    ax.add_patch(plt.Rectangle((21.1, 83.5), 57.8, 16.5,
                 facecolor="#ffffff", alpha=0.025, edgecolor="none", zorder=1))

    def l(xs, ys, lw=1.2, a=0.75, ls="-"):
        ax.plot(xs, ys, color="white", lw=lw, alpha=a, ls=ls,
                zorder=2, solid_capstyle="round")

    y0 = 50 if half else 0
    # Outer boundary
    l([0, 100, 100, 0, 0], [y0, y0, 100, 100, y0], lw=1.9, a=0.88)
    if not half:
        l([0, 100], [50, 50], lw=1.0, a=0.40, ls="--")
        ax.add_patch(plt.Circle((50, 50), 9.15 * 100 / 105,
                                color="white", fill=False, lw=0.9, alpha=0.38, zorder=2))
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
    ax.add_patch(matplotlib.patches.Arc(
        (50, 83.5), 18 * 100 / 105, 18, angle=0, theta1=0, theta2=180,
        color="white", lw=0.9, alpha=0.38, zorder=2))
    if not half:
        ax.add_patch(matplotlib.patches.Arc(
            (50, 16.5), 18 * 100 / 105, 18, angle=0, theta1=180, theta2=360,
            color="white", lw=0.9, alpha=0.32, zorder=2))
    # Penalty spots
    ax.plot(50, 89, "o", color="white", ms=1.8, alpha=0.55, zorder=2)

    xl = -3; xr = 103
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

    hn, an   = info["home_name"], info["away_name"]
    hid, aid = info["home_id"],   info["away_id"]

    ev = events[events[["x", "y"]].notna().all(axis=1)].copy()
    cw = 100 / N_COLS
    rh = 100 / N_ROWS

    CONTESTED = "#5a5f6b"
    ALPHA_DOM  = 0.68
    ALPHA_CONT = 0.42

    for ci in range(N_COLS):
        for ri in range(N_ROWS):
            x0, x1 = ci * cw, (ci + 1) * cw
            y0, y1 = ri * rh, (ri + 1) * rh
            zone = ev[(ev["x"] >= x0) & (ev["x"] < x1) &
                      (ev["y"] >= y0) & (ev["y"] < y1)]
            h_n = int((zone["team_id"] == hid).sum())
            a_n = int((zone["team_id"] == aid).sum())
            tot = (h_n + a_n) or 1
            hr  = h_n / tot

            if hr > 0.55:
                col, alpha = C_RED, ALPHA_DOM
            elif hr < 0.45:
                col, alpha = C_BLUE, ALPHA_DOM
            else:
                col, alpha = CONTESTED, ALPHA_CONT

            ax.add_patch(plt.Rectangle(
                (x0 + 0.4, y0 + 0.4), cw - 0.8, rh - 0.8,
                facecolor=col, alpha=alpha, edgecolor="none", zorder=2,
                transform=ax.transData))

    # Legend
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_RED,    alpha=0.78, label=f"{hn[:12]}  (>55%)"),
        mpatches.Patch(facecolor=CONTESTED,alpha=0.65, label="Contested  (45–55%)"),
        mpatches.Patch(facecolor=C_BLUE,   alpha=0.78, label=f"{an[:12]}  (>55%)"),
    ], fontsize=7.5, ncol=3, facecolor=BG_MID, edgecolor=GRID_COL,
       labelcolor=TEXT_MAIN, loc="lower center",
       bbox_to_anchor=(0.50, -0.07), framealpha=0.92)

    # Attacking direction labels
    ax.text(1, -5.5, f"← {hn} Attacks",  ha="left",  va="top",
            color=C_RED,  fontsize=7, fontweight="bold")
    ax.text(99, -5.5, f"{an} Attacks →", ha="right", va="top",
            color=C_BLUE, fontsize=7, fontweight="bold")


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

    PBX1, PBX2 = 21.1, 78.9   # y on vertical pitch (= event.y transposed)
    PBY1       = 83.5           # x on vertical pitch (= event.x)

    def in_box(ex, ey):
        return ex >= PBY1 and PBX1 <= ey <= PBX2

    def _vx(ey): return float(ey)            # event.y  → ax x
    def _vy(ex): return float(ex)            # event.x  → ax y

    # Passes
    passes = events[
        (events["is_pass"] == True) &
        (events["team_id"] == tid) &
        (events["outcome"] == "Successful") &
        events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    pass_entries = passes[
        passes.apply(lambda r: not in_box(r["x"], r["y"]) and
                               in_box(r["end_x"], r["end_y"]), axis=1)
    ]

    # Carries (type == "Carry" if present)
    carry_entries = pd.DataFrame()
    if "type" in events.columns:
        carries = events[
            (events["type"] == "Carry") &
            (events["team_id"] == tid) &
            events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        ].copy()
        carry_entries = carries[
            carries.apply(lambda r: not in_box(r["x"], r["y"]) and
                                    in_box(r["end_x"], r["end_y"]), axis=1)
        ]

    # Draw passes
    for _, r in pass_entries.iterrows():
        ax.annotate("", xy=(_vx(r["end_y"]), _vy(r["end_x"])),
                    xytext=(_vx(r["y"]), _vy(r["x"])),
                    arrowprops=dict(arrowstyle="-|>", color=tc,
                                    lw=1.5, alpha=0.88, mutation_scale=8), zorder=5)
    # Draw carries
    for _, r in carry_entries.iterrows():
        ax.annotate("", xy=(_vx(r["end_y"]), _vy(r["end_x"])),
                    xytext=(_vx(r["y"]), _vy(r["x"])),
                    arrowprops=dict(arrowstyle="-|>", color="#c084fc",
                                    lw=1.3, alpha=0.82,
                                    linestyle="dashed", mutation_scale=7), zorder=5)

    # End-point dots inside box
    for _, r in pass_entries.iterrows():
        ax.scatter(_vx(r["end_y"]), _vy(r["end_x"]),
                   c=tc, s=22, edgecolors="white", lw=0.6, zorder=6)
    for _, r in carry_entries.iterrows():
        ax.scatter(_vx(r["end_y"]), _vy(r["end_x"]),
                   c="#c084fc", s=18, edgecolors="white", lw=0.5, zorder=6)

    # Entry-side breakdown (left / mid / right based on end_y)
    n_left  = int(((pass_entries["end_y"] < 35)).sum() +
                  (len(carry_entries) and (carry_entries["end_y"] < 35).sum() or 0))
    n_right = int(((pass_entries["end_y"] > 65)).sum() +
                  (len(carry_entries) and (carry_entries["end_y"] > 65).sum() or 0))
    n_mid   = (len(pass_entries) + len(carry_entries)) - n_left - n_right

    for xp, lbl_, val in [(15, "Left", n_left), (50, "Mid", n_mid), (85, "Right", n_right)]:
        ax.text(xp, 48.5, f"{lbl_}\n{val}",
                ha="center", va="top", color=tc,
                fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor=BG_DARK,
                          alpha=0.80, edgecolor="none"))

    # Stats line
    n_p = len(pass_entries); n_c = len(carry_entries)
    ax.text(50, 106, f"Total: {n_p + n_c}   By Pass: {n_p}   By Carry: {n_c}",
            ha="center", va="bottom", color=TEXT_DIM, fontsize=7)

    ax.legend(handles=[
        Line2D([0],[0], color=tc,       lw=1.5, label=f"Pass ({n_p})"),
        Line2D([0],[0], color="#c084fc", lw=1.3, linestyle="dashed", label=f"Carry ({n_c})"),
    ], fontsize=7, facecolor=BG_MID, edgecolor="none",
       labelcolor="white", loc="lower right", framealpha=0.85)


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
            (events["type"] == "Carry") &
            (events["team_id"] == tid) &
            events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        ].copy()

    if carries.empty:
        # Fallback: approximate from consecutive same-player touch sequences
        ax.text(50, 50, "No carry data\nin this match",
                ha="center", va="center", color=TEXT_DIM,
                fontsize=8, style="italic")
        return

    # Progressive: forward distance ≥ 9.5 units, not from own defensive third
    carries["fwd"] = carries["end_x"] - carries["x"]
    prog = carries[(carries["fwd"] >= 9.5) & (carries["x"] >= 33)].copy()

    if prog.empty:
        ax.text(50, 50, "No progressive\ncarries found",
                ha="center", va="center", color=TEXT_DIM, fontsize=8, style="italic")
        return

    def _vx(ey): return float(ey)
    def _vy(ex): return float(ex)

    # Zone colours by origin third
    def zone_col(ex):
        if ex < 50:   return "#64748b"
        if ex < 66:   return C_GOLD
        return C_GREEN

    for _, r in prog.iterrows():
        col_ = zone_col(r["x"])
        # Glow
        ax.annotate("", xy=(_vx(r["end_y"]), _vy(r["end_x"])),
                    xytext=(_vx(r["y"]), _vy(r["x"])),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                    lw=3.5, alpha=0.06, mutation_scale=10), zorder=3)
        ax.annotate("", xy=(_vx(r["end_y"]), _vy(r["end_x"])),
                    xytext=(_vx(r["y"]), _vy(r["x"])),
                    arrowprops=dict(arrowstyle="-|>", color=col_,
                                    lw=1.4, alpha=0.78,
                                    linestyle=(0, (4, 2)), mutation_scale=7), zorder=4)
        ax.scatter(_vx(r["y"]), _vy(r["x"]),
                   c=col_, s=12, alpha=0.75, edgecolors="none", zorder=5)

    # Top-5 carries — player label
    if "player" in prog.columns:
        top_player = prog.groupby("player").size().idxmax()
        top_n      = prog.groupby("player").size().max()
        ax.text(50, 106, f"Most by: {_short(top_player)} ({top_n})",
                ha="center", va="bottom", color="#facc15",
                fontsize=7.5, fontweight="bold")

    # From left / mid / right (based on event.y)
    n_left  = int((prog["y"] < 33).sum())
    n_right = int((prog["y"] > 67).sum())
    n_mid   = len(prog) - n_left - n_right
    tot     = len(prog)

    for xp, lbl_, n, col_ in [
        (10, "From Left", n_left,  tc),
        (50, "From Mid",  n_mid,   TEXT_DIM),
        (90, "From Right",n_right, tc),
    ]:
        ax.text(xp, -3.5, f"{lbl_}\n{n}",
                ha="center", va="top", color=col_,
                fontsize=7, fontweight="bold")

    ax.legend(handles=[
        mpatches.Patch(facecolor="#64748b", label="Own Half"),
        mpatches.Patch(facecolor=C_GOLD,   label="Mid Third"),
        mpatches.Patch(facecolor=C_GREEN,  label="Final Third"),
    ], fontsize=6.5, facecolor=BG_MID, edgecolor="none",
       labelcolor="white", loc="lower right",
       ncol=1, framealpha=0.85)


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

    R40 = 40 / 105 * 100         # 40m in WhoScored x-units ≈ 38.1
    GCX = 100                    # goal centre x in WhoScored (opponent)
    GCY = 50

    # Highlight 40m circle — light fill
    circle = plt.Circle(
        (GCY, GCX), R40,          # (ax_x, ax_y) after transpose
        facecolor=tc, alpha=0.10,
        edgecolor=tc, lw=1.2, linestyle="--",
        zorder=2)
    ax.add_patch(circle)

    # High-turnover event types
    HT_TYPES = {"Interception", "BallRecovery", "Tackle", "BlockedShot", "Clearance"}
    ht_events = events[
        (events["team_id"] == tid) &
        (events["type"].isin(HT_TYPES)) &
        events[["x", "y"]].notna().all(axis=1)
    ].copy()

    # Filter: inside 40m zone
    ht_events["dist_goal"] = np.sqrt(
        ((ht_events["x"] - GCX) ** 2) + ((ht_events["y"] - GCY) ** 2))
    ht_high = ht_events[ht_events["dist_goal"] <= R40].copy()

    n_total = len(ht_high)

    # Scatter
    if not ht_high.empty:
        ax.scatter(ht_high["y"], ht_high["x"],
                   c=tc, s=45, edgecolors="white", lw=0.8,
                   alpha=0.92, zorder=6)

    # Count led to shot (within next 3 minutes of play → approximate:
    # same period, minute within +3 of turnover)
    led_shot = led_goal = 0
    if not ht_high.empty and "minute" in events.columns:
        ev_sorted = events.sort_values(["minute", "second"]).reset_index(drop=True)
        for _, row in ht_high.iterrows():
            min_ = row.get("minute", 0)
            pid  = row.get("period_code", "")
            window = ev_sorted[
                (ev_sorted["team_id"] == tid) &
                (ev_sorted["minute"] >= min_) &
                (ev_sorted["minute"] <= min_ + 3) &
                (ev_sorted["period_code"] == pid)
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
        ax.fill(hx, hy, facecolor=col, alpha=0.92,
                edgecolor="white", lw=1.0, zorder=8,
                transform=ax.transAxes)
        ax.text(cx, cy + 0.025, str(val), ha="center", va="center",
                color="white", fontsize=11, fontweight="bold",
                transform=ax.transAxes, zorder=9)
        ax.text(cx, cy - 0.030, txt, ha="center", va="center",
                color="white", fontsize=6.5, fontweight="bold",
                transform=ax.transAxes, zorder=9)

    _hex(ax, 0.50, 0.26,  "Total",         n_total,   tc)
    _hex(ax, 0.35, 0.14,  "Led to\nShot",  led_shot,  tc, size=0.075)
    _hex(ax, 0.65, 0.14,  "Led to\nGoal",  led_goal,  tc, size=0.075)

    # 40m radius annotation
    ax.text(50, 61, "← 40m radius →", ha="center", va="center",
            color=tc, fontsize=6.5, alpha=0.65, fontstyle="italic")


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
        (events["is_pass"] == True) &
        (events["team_id"] == tid) &
        (events["outcome"] == "Successful") &
        events[["end_x", "end_y"]].notna().all(axis=1)
    ].copy()

    if passes.empty:
        ax.text(50, 50, "No pass data", ha="center", va="center",
                color=TEXT_DIM, fontsize=8, style="italic")
        return

    N_COLS, N_ROWS = 5, 6   # width × height
    cw = 100 / N_COLS
    rh = 100 / N_ROWS
    total = len(passes)

    # Build percentage grid
    grid = np.zeros((N_ROWS, N_COLS))
    for ci in range(N_COLS):
        for ri in range(N_ROWS):
            ex0, ex1 = (N_ROWS - 1 - ri) * rh, (N_ROWS - ri) * rh   # top = high x
            ey0, ey1 = ci * cw, (ci + 1) * cw
            n = int(passes[(passes["end_x"] >= ex0) & (passes["end_x"] < ex1) &
                            (passes["end_y"] >= ey0) & (passes["end_y"] < ey1)].shape[0])
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
            ax.add_patch(plt.Rectangle(
                (ax_x0 + 0.5, ax_y0 + 0.5), cw - 1, rh - 1,
                facecolor=tc, alpha=alpha,
                edgecolor="#2d3748", lw=0.5, zorder=2))
            if pct >= 0.5:
                txt_col = "white"
                ax.text(ax_x0 + cw / 2, ax_y0 + rh / 2 + rh * 0.12,
                        f"{pct:.1f}%",
                        ha="center", va="center", color=txt_col,
                        fontsize=9.5, fontweight="bold", zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
                # عدد التمريرات تحت النسبة
                n_passes = int(round(pct * total / 100))
                ax.text(ax_x0 + cw / 2, ax_y0 + rh / 2 - rh * 0.22,
                        f"({n_passes})",
                        ha="center", va="center", color="#cbd5e1",
                        fontsize=7.5, zorder=4,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    # Scatter actual pass destinations as dim dots
    ax.scatter(passes["end_y"], passes["end_x"],
               c="#94a3b8", s=4, alpha=0.25, edgecolors="none", zorder=3)

    ax.text(50, 106, f"Total Passes: {total}",
            ha="center", va="bottom", color=TEXT_DIM, fontsize=7)



# ══════════════════════════════════════════════════════════════════════
#  TACTICAL ANALYSIS ENGINE  (fully automatic — no API needed)
# ══════════════════════════════════════════════════════════════════════
def _collect_match_stats(info, events, xg_data):
    """Extract all key stats into a flat dict."""
    hn, an   = info["home_name"], info["away_name"]
    hid, aid = info["home_id"],   info["away_id"]

    def _team(tid, tname):
        ev = events[events["team_id"] == tid]

        goals     = int(ev["is_goal"].sum())    if "is_goal" in ev.columns else 0
        shots_ev  = ev[ev["is_shot"] == True]   if "is_shot" in ev.columns else pd.DataFrame()
        n_shots   = len(shots_ev)
        on_tgt    = int(shots_ev[shots_ev["type"].isin(["Goal","SavedShot"])].shape[0]) \
                    if not shots_ev.empty and "type" in shots_ev.columns else 0

        passes    = ev[ev["is_pass"] == True]   if "is_pass" in ev.columns else pd.DataFrame()
        p_succ    = int(passes[passes["outcome"] == "Successful"].shape[0]) if not passes.empty else 0
        p_total   = len(passes)
        p_pct     = round(p_succ / p_total * 100) if p_total else 0

        prog_p    = int(passes[(passes["end_x"] - passes["x"] >= 9.5) &
                               (passes["x"] >= 33)].shape[0]) \
                    if not passes.empty and "end_x" in passes.columns else 0

        fwd_p     = int(passes[passes["end_x"] > passes["x"]].shape[0]) \
                    if not passes.empty and "end_x" in passes.columns else 0
        fwd_pct   = round(fwd_p / p_total * 100) if p_total else 0

        back_p    = int(passes[passes["end_x"] < passes["x"]].shape[0]) \
                    if not passes.empty and "end_x" in passes.columns else 0

        crosses_ev = ev[ev["type"] == "Cross"] if "type" in ev.columns else pd.DataFrame()
        cross_succ = int(crosses_ev[crosses_ev["outcome"] == "Successful"].shape[0]) if not crosses_ev.empty else 0
        cross_pct  = round(cross_succ / len(crosses_ev) * 100) if len(crosses_ev) else 0

        def_ev    = ev[ev["type"].isin(["Tackle","Interception","BallRecovery",
                                        "Clearance","BlockedShot"])] \
                    if "type" in ev.columns else pd.DataFrame()
        tackles   = int(ev[ev["type"] == "Tackle"].shape[0])      if "type" in ev.columns else 0
        intercept = int(ev[ev["type"] == "Interception"].shape[0]) if "type" in ev.columns else 0
        clearance = int(ev[ev["type"] == "Clearance"].shape[0])    if "type" in ev.columns else 0
        blocked   = int(ev[ev["type"] == "BlockedShot"].shape[0])  if "type" in ev.columns else 0
        recoveries= int(ev[ev["type"] == "BallRecovery"].shape[0]) if "type" in ev.columns else 0

        fouls     = int(ev[ev["type"] == "Foul"].shape[0])         if "type" in ev.columns else 0

        ht_ev     = def_ev[np.sqrt(((def_ev["x"] - 100)**2) +
                                   ((def_ev["y"] - 50)**2)) <= 38] \
                    if not def_ev.empty and "x" in def_ev.columns else pd.DataFrame()

        z14       = int(ev[(ev["x"].between(66, 83)) &
                           (ev["y"].between(33, 67))].shape[0]) if "x" in ev.columns else 0

        touches   = int(ev[ev["type"].isin(["Pass","TakeOn","Carry","BallRecovery",
                                             "Tackle","Interception","Clearance"]
                           )].shape[0]) if "type" in ev.columns else 0

        # touches by third
        t_def     = int(ev[ev["x"] < 33]["type"].count())          if "x" in ev.columns else 0
        t_mid     = int(ev[ev["x"].between(33, 67)]["type"].count())if "x" in ev.columns else 0
        t_att     = int(ev[ev["x"] > 67]["type"].count())          if "x" in ev.columns else 0
        tot_thirds= (t_def + t_mid + t_att) or 1
        t_def_pct = round(t_def / tot_thirds * 100)
        t_mid_pct = round(t_mid / tot_thirds * 100)
        t_att_pct = round(t_att / tot_thirds * 100)

        xg_t = xg_data.get(tname, {})

        return {
            "goals":          goals,
            "shots":          xg_t.get("shots", n_shots),
            "on_target":      xg_t.get("on_target", on_tgt),
            "xG":             round(xg_t.get("xG", 0), 2),
            "xGoT":           round(xg_t.get("xGoT", 0), 2),
            "passes_total":   p_total,
            "pass_accuracy":  p_pct,
            "prog_passes":    prog_p,
            "fwd_passes":     fwd_p,
            "fwd_pct":        fwd_pct,
            "back_passes":    back_p,
            "crosses_total":  len(crosses_ev),
            "crosses_succ":   cross_succ,
            "cross_pct":      cross_pct,
            "touches":        touches,
            "touch_def_pct":  t_def_pct,
            "touch_mid_pct":  t_mid_pct,
            "touch_att_pct":  t_att_pct,
            "zone14_touches": z14,
            "defensive_acts": len(def_ev),
            "tackles":        tackles,
            "interceptions":  intercept,
            "clearances":     clearance,
            "blocked_shots":  blocked,
            "recoveries":     recoveries,
            "fouls":          fouls,
            "high_turnovers": len(ht_ev),
        }

    return {
        "home": _team(hid, hn),
        "away": _team(aid, an),
        "home_name": hn,
        "away_name": an,
        "score":       info.get("score", "? - ?"),
        "venue":       info.get("venue", ""),
        "date":        info.get("date", ""),
        "competition": info.get("competition", ""),
    }


def _w(team, other, key, higher_is_better=True):
    """Return 'dominant'/'stronger'/'weaker' label based on stat comparison."""
    tv = team.get(key, 0)
    ov = other.get(key, 0)
    if tv == ov:    return "equal"
    if higher_is_better:
        return "superior" if tv > ov else "inferior"
    return "superior" if tv < ov else "inferior"


def generate_tactical_analysis(info, events, xg_data):
    """
    Build a full publication-ready tactical report purely from match stats.
    No external API required.
    """
    stats = _collect_match_stats(info, events, xg_data)
    hn, an = stats["home_name"], stats["away_name"]
    h, a   = stats["home"], stats["away"]

    score_parts = stats["score"].split("-")
    hg = int(score_parts[0].strip()) if len(score_parts) >= 2 else h["goals"]
    ag = int(score_parts[1].strip()) if len(score_parts) >= 2 else a["goals"]

    winner  = hn if hg > ag else (an if ag > hg else None)
    loser   = an if hg > ag else (hn if ag > hg else None)
    margin  = abs(hg - ag)
    draw    = hg == ag

    def _dom(hv, av, unit=""):
        """Return e.g. 'Arsenal (28) vs Chelsea (15)' string."""
        return f"{hn} ({hv}{unit}) vs {an} ({av}{unit})"

    def _better(hv, av, hn_=hn, an_=an):
        return hn_ if hv > av else (an_ if av > hv else "Both sides equally")

    def _pct_diff(a_, b_):
        if b_ == 0: return 0
        return round((a_ - b_) / b_ * 100)

    # ── MATCH OVERVIEW ─────────────────────────────────────────────
    if draw:
        result_line = (f"The match between {hn} and {an} ended in a {hg}–{ag} draw, "
                       "a scoreline that suggested relative parity between the two sides over 90 minutes.")
    else:
        result_line = (f"{winner} defeated {loser} {hg}–{ag}"
                       f"{' in a comprehensive victory' if margin >= 3 else ' in a hard-fought contest' if margin == 1 else ''}.")

    xg_narrative = ""
    if h["xG"] > 0 or a["xG"] > 0:
        xg_winner = _better(h["xG"], a["xG"])
        xg_narrative = (f" The xG figures — {hn}: {h['xG']}, {an}: {a['xG']} — "
                        f"indicate that {xg_winner} created the higher-quality opportunities.")

    pass_narrative = (f" In terms of ball circulation, {hn} recorded {h['pass_accuracy']}% pass accuracy "
                      f"versus {an}'s {a['pass_accuracy']}%, "
                      f"with {_better(h['prog_passes'], a['prog_passes'])} generating more forward momentum "
                      f"through progressive passes ({_dom(h['prog_passes'], a['prog_passes'])}).")

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
    acc_h  = round(h["on_target"] / h["shots"] * 100) if h["shots"] else 0
    acc_a  = round(a["on_target"] / a["shots"] * 100) if a["shots"] else 0

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

    ht_winner = _better(h['high_turnovers'], a['high_turnovers'])
    ht_max    = max(h['high_turnovers'], a['high_turnovers'])
    ht_sentence = (
        f"The pressing game was a decisive factor — {ht_winner}'s {ht_max} high turnovers "
        f"disrupted the opponent and created additional transitional opportunities."
        if ht_max > 3 else
        "High pressing was not a decisive factor, with both teams showing adequate composure under pressure."
    )
    ultimately = (
        f"the result accurately reflects the statistical superiority of {winner}"
        if not draw else
        "neither team managed to translate statistical advantages into a winning goal"
    )
    match_type = (
        "a convincing display of modern tactical football." if not draw and margin >= 3
        else "a tight tactical contest settled by fine margins." if not draw
        else "a balanced tactical encounter."
    )
    pp_diff = abs(h['prog_passes'] - a['prog_passes'])
    pp_note = (f"ball progression ({_dom(h['prog_passes'], a['prog_passes'])} progressive passes)"
               if pp_diff > 5 else "a competitive midfield battle")
    phases_intro = "by " + winner + " " if not draw else ""
    phases_kind  = "multiple" if not draw and margin >= 2 else "critical"
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
        "MATCH OVERVIEW":           overview,
        "xG & SHOOTING ANALYSIS":   xg_shooting,
        "PASSING & BALL PROGRESSION": passing,
        "PASS NETWORKS":            pass_networks,
        "xT (EXPECTED THREAT)":     xt_analysis,
        "SHOT COMPARISON":          shot_cmp,
        "DANGER CREATION":          danger,
        "ZONE 14 & HALF-SPACES":    z14_analysis,
        "TERRITORIAL CONTROL":      territorial,
        "POSSESSION & TOUCHES":     poss_analysis,
        "PASS MAP BY THIRD":        pass_thirds,
        "CROSSES":                  cross_analysis,
        "DEFENSIVE HEATMAP":        def_heatmap,
        "DEFENSIVE SUMMARY":        def_summary,
        "AVERAGE POSITIONS":        avg_pos,
        "DOMINATING ZONE":          dom_zone,
        "BOX ENTRIES":              box_entries,
        "HIGH TURNOVERS":           high_to,
        "PASS TARGET ZONES":        pass_target,
        "TACTICAL VERDICT":         verdict,
    }



# ══════════════════════════════════════════════════════════════════════
#  PDF BUILDER  (tactical report)
# ══════════════════════════════════════════════════════════════════════
def _parse_scoreline(info, xg_data):
    """Return home and away score strings, even if score text is missing or uses fancy dashes."""
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
        (events["is_pass"] == True) &
        (events["team_id"] == tid) &
        events[["x", "end_x"]].notna().all(axis=1)
    ].copy()

    profile = {
        "total": len(passes),
        "def": 0, "mid": 0, "att": 0,
        "succ_def": 0, "succ_mid": 0, "succ_att": 0,
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
        profile[f"succ_{key}"] = int((df_["outcome"] == "Successful").sum()) if "outcome" in df_.columns else 0
    return profile


def _progressive_profile(events, tid):
    passes = events[
        (events["is_pass"] == True) &
        (events["team_id"] == tid) &
        (events["outcome"] == "Successful") &
        events[["x", "end_x"]].notna().all(axis=1)
    ].copy()
    passes = passes[(passes["end_x"] - passes["x"]) >= 10].copy() if not passes.empty else passes

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
        (events["is_cross"] == True) &
        (events["team_id"] == tid) &
        events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    if crosses.empty:
        return {"total": 0, "succ": 0, "left": 0, "middle": 0, "right": 0}

    left = int((crosses["y"] < 40).sum())
    right = int((crosses["y"] > 60).sum())
    middle = len(crosses) - left - right

    return {
        "total": len(crosses),
        "succ": int((crosses["outcome"] == "Successful").sum()) if "outcome" in crosses.columns else 0,
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
        (events["is_pass"] == True) &
        (events["team_id"] == tid) &
        (events["outcome"] == "Successful") &
        events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    ].copy()
    pass_entries = passes[
        passes.apply(lambda r: (not in_box(r["x"], r["y"])) and in_box(r["end_x"], r["end_y"]), axis=1)
    ] if not passes.empty else passes

    carry_entries = pd.DataFrame()
    if "type" in events.columns:
        carries = events[
            (events["type"] == "Carry") &
            (events["team_id"] == tid) &
            events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        ].copy()
        carry_entries = carries[
            carries.apply(lambda r: (not in_box(r["x"], r["y"])) and in_box(r["end_x"], r["end_y"]), axis=1)
        ] if not carries.empty else carries

    if pass_entries.empty and carry_entries.empty:
        return {"total": 0, "pass": 0, "carry": 0, "left": 0, "middle": 0, "right": 0}

    end_y = pd.concat([
        pass_entries["end_y"] if not pass_entries.empty else pd.Series(dtype=float),
        carry_entries["end_y"] if not carry_entries.empty else pd.Series(dtype=float),
    ], ignore_index=True)

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
        (events["team_id"] == tid) &
        (events["type"].isin(ht_types)) &
        events[["x", "y"]].notna().all(axis=1)
    ].copy()
    if ht_events.empty:
        return {"total": 0, "led_shot": 0, "led_goal": 0}

    ht_events["dist_goal"] = np.sqrt(((ht_events["x"] - gcx) ** 2) + ((ht_events["y"] - gcy) ** 2))
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
                (ev_sorted["team_id"] == tid) &
                (ev_sorted["minute"] >= minute) &
                (ev_sorted["minute"] <= minute + 3) &
                (ev_sorted["period_code"] == period_code)
            ]
            if "is_shot" in window.columns and window["is_shot"].any():
                led_shot += 1
            if "is_goal" in window.columns and window["is_goal"].any():
                led_goal += 1

    return {"total": len(ht_high), "led_shot": led_shot, "led_goal": led_goal}


def _build_visual_catalog(info):
    hn, an = info["home_name"], info["away_name"]
    return [
        {"idx": 1,  "section": "shared", "team": None,   "kind": "shared_xg_flow",           "title": "xG Flow"},
        {"idx": 2,  "section": "home",   "team": "home", "kind": "team_shot_map",            "title": f"{hn} Shot Map"},
        {"idx": 3,  "section": "away",   "team": "away", "kind": "team_shot_map",            "title": f"{an} Shot Map"},
        {"idx": 4,  "section": "shared", "team": None,   "kind": "shared_shot_breakdown",    "title": "Shot Breakdown and Goals"},
        {"idx": 5,  "section": "home",   "team": "home", "kind": "team_pass_network",        "title": f"{hn} Pass Network"},
        {"idx": 6,  "section": "away",   "team": "away", "kind": "team_pass_network",        "title": f"{an} Pass Network"},
        {"idx": 7,  "section": "home",   "team": "home", "kind": "team_xt_map",              "title": f"{hn} xT Map"},
        {"idx": 8,  "section": "away",   "team": "away", "kind": "team_xt_map",              "title": f"{an} xT Map"},
        {"idx": 9,  "section": "shared", "team": None,   "kind": "shared_shot_comparison",   "title": "Shot Comparison"},
        {"idx": 10, "section": "home",   "team": "home", "kind": "team_danger_creation",     "title": f"{hn} Danger Creation"},
        {"idx": 11, "section": "away",   "team": "away", "kind": "team_danger_creation",     "title": f"{an} Danger Creation"},
        {"idx": 12, "section": "shared", "team": None,   "kind": "shared_gk_saves",          "title": "Goalkeeper Saves"},
        {"idx": 13, "section": "shared", "team": None,   "kind": "shared_xg_tiles",          "title": "xG and xGoT Summary"},
        {"idx": 14, "section": "home",   "team": "home", "kind": "team_zone14",              "title": f"{hn} Zone 14 and Half-Spaces"},
        {"idx": 15, "section": "away",   "team": "away", "kind": "team_zone14",              "title": f"{an} Zone 14 and Half-Spaces"},
        {"idx": 16, "section": "shared", "team": None,   "kind": "shared_match_stats",       "title": "Match Statistics"},
        {"idx": 17, "section": "shared", "team": None,   "kind": "shared_territorial",       "title": "Territorial Control"},
        {"idx": 18, "section": "shared", "team": None,   "kind": "shared_touches",           "title": "Ball Touches"},
        {"idx": 19, "section": "home",   "team": "home", "kind": "team_pass_thirds",         "title": f"{hn} Pass Map by Third"},
        {"idx": 20, "section": "away",   "team": "away", "kind": "team_pass_thirds",         "title": f"{an} Pass Map by Third"},
        {"idx": 21, "section": "shared", "team": None,   "kind": "shared_xt_per_minute",     "title": "xT per Minute"},
        {"idx": 22, "section": "home",   "team": "home", "kind": "team_progressive_passes",  "title": f"{hn} Progressive Passes"},
        {"idx": 23, "section": "away",   "team": "away", "kind": "team_progressive_passes",  "title": f"{an} Progressive Passes"},
        {"idx": 24, "section": "home",   "team": "home", "kind": "team_crosses",             "title": f"{hn} Crosses"},
        {"idx": 25, "section": "away",   "team": "away", "kind": "team_crosses",             "title": f"{an} Crosses"},
        {"idx": 26, "section": "home",   "team": "home", "kind": "team_def_heatmap",         "title": f"{hn} Defensive Actions"},
        {"idx": 27, "section": "away",   "team": "away", "kind": "team_def_heatmap",         "title": f"{an} Defensive Actions"},
        {"idx": 28, "section": "shared", "team": None,   "kind": "shared_def_summary",       "title": "Defensive Summary"},
        {"idx": 29, "section": "home",   "team": "home", "kind": "team_average_positions",   "title": f"{hn} Average Positions"},
        {"idx": 30, "section": "away",   "team": "away", "kind": "team_average_positions",   "title": f"{an} Average Positions"},
        {"idx": 31, "section": "shared", "team": None,   "kind": "shared_dominating_zone",   "title": "Dominating Zone"},
        {"idx": 32, "section": "home",   "team": "home", "kind": "team_box_entries",         "title": f"{hn} Box Entries"},
        {"idx": 33, "section": "away",   "team": "away", "kind": "team_box_entries",         "title": f"{an} Box Entries"},
        {"idx": 34, "section": "home",   "team": "home", "kind": "team_high_turnovers",      "title": f"{hn} High Turnovers"},
        {"idx": 35, "section": "away",   "team": "away", "kind": "team_high_turnovers",      "title": f"{an} High Turnovers"},
        {"idx": 36, "section": "home",   "team": "home", "kind": "team_pass_target_zones",   "title": f"{hn} Pass Target Zones"},
        {"idx": 37, "section": "away",   "team": "away", "kind": "team_pass_target_zones",   "title": f"{an} Pass Target Zones"},
    ]


def _shared_section_summary(info, stats, events):
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    hg, ag = _parse_scoreline(info, {hn: {"goals": h["goals"]}, an: {"goals": a["goals"]}})

    opener = (
        f"{hn} and {an} finished level at {hg}-{ag}. The comparison pages open with the numbers that shaped that balance."
        if str(hg) == str(ag)
        else f"{hn} beat {an} {hg}-{ag}. The shared pages explain where the overall edge came from before the report moves into each team on its own."
    )
    xg_leader = _leader_name(h["xG"], a["xG"], hn, an)
    prog_leader = _leader_name(h["prog_passes"], a["prog_passes"], hn, an)
    xt_leader = _leader_name(_xt_total(events, info["home_id"]), _xt_total(events, info["away_id"]), hn, an)

    return (
        f"{opener} {xg_leader} produced the stronger chance profile on xG, while {prog_leader} carried more of the forward progression through progressive passing. "
        f"The shared charts also show how territory, defensive workload and threat changed over the game rather than in a single phase. "
        f"Use this section to read the match at a high level first, then follow the report into the home and away details. {xt_leader} also had the stronger overall threat return when possession started to turn into real danger."
    )


def _team_section_summary(side, info, stats, events):
    team = stats[side]
    other_side = "away" if side == "home" else "home"
    opp = stats[other_side]
    team_name = info[f"{side}_name"]
    opp_name = info[f"{other_side}_name"]
    xt_total = _xt_total(events, info[f"{side}_id"])

    return (
        f"This section isolates {team_name} so the report can read the team without the noise of direct comparison charts. "
        f"Across the match, {team_name} finished with {team['shots']} shots, {team['on_target']} on target and {team['xG']} xG, alongside {team['prog_passes']} progressive passes and {team['defensive_acts']} defensive actions. "
        f"Compared with {opp_name}, the next pages should tell you whether their shape was built on controlled circulation, direct progression, wide delivery or defensive work. "
        f"Their total xT return came out at {_fmt_num(xt_total, 2)}, which is a useful guide for judging how much of the ball actually became threat."
    )


def _visual_tactical_note(meta, info, events, xg_data, stats):
    hn, an = info["home_name"], info["away_name"]
    h, a = stats["home"], stats["away"]
    hid, aid = info["home_id"], info["away_id"]
    hg, ag = _parse_scoreline(info, xg_data)
    kind = meta["kind"]

    if meta["team"] == "home":
        side_key, other_key = "home", "away"
        team_name, opp_name = hn, an
        tid = hid
    elif meta["team"] == "away":
        side_key, other_key = "away", "home"
        team_name, opp_name = an, hn
        tid = aid
    else:
        side_key = other_key = team_name = opp_name = tid = None

    if side_key:
        team = stats[side_key]
        opp = stats[other_key]
        team_xt = _xt_total(events, info[f"{side_key}_id"])
        opp_xt = _xt_total(events, info[f"{other_key}_id"])
    else:
        team = opp = None
        team_xt = opp_xt = None

    if kind == "shared_xg_flow":
        statline = f"Score {hg}-{ag}  |  xG {h['xG']}-{a['xG']}  |  Progressive passes {h['prog_passes']}-{a['prog_passes']}"
        note = (
            f"The xG flow gives the cleanest read on how the {hg}-{ag} score developed. {_leader_name(h['xG'], a['xG'], hn, an)} finished with the stronger chance profile on overall xG, which means the better openings were not just a matter of shot volume. "
            f"Set against the progressive-pass split of {h['prog_passes']} to {a['prog_passes']}, the chart helps separate simple possession from moments that actually moved the match toward goal."
        )
    elif kind == "shared_shot_breakdown":
        statline = f"Shots {h['shots']}-{a['shots']}  |  On target {h['on_target']}-{a['on_target']}  |  Goals {h['goals']}-{a['goals']}"
        note = (
            f"This page breaks the attack into output and outcome. {hn} took {h['shots']} shots and scored {h['goals']}, while {an} finished with {a['shots']} shots and {a['goals']} goals. "
            f"Read the on-target numbers and the blocked-shot totals together here: they show whether the attacks reached clean striking lanes or kept running into traffic before the final touch."
        )
    elif kind == "shared_shot_comparison":
        statline = f"xG {h['xG']}-{a['xG']}  |  Shots {h['shots']}-{a['shots']}  |  On target {h['on_target']}-{a['on_target']}"
        note = (
            f"The comparison tiles make the finishing profile easier to read at a glance. {_leader_name(h['xG'], a['xG'], hn, an)} paired the stronger xG return with the better shot profile, which usually means they reached higher-value spaces more often. "
            f"When shot volume and shot quality move in the same direction, the attacking edge is usually real rather than cosmetic."
        )
    elif kind == "shared_gk_saves":
        hs = xg_data.get(hn, {}).get("saved", 0)
        a_s = xg_data.get(an, {}).get("saved", 0)
        statline = f"Saves {hs}-{a_s}  |  xGoT {h['xGoT']}-{a['xGoT']}"
        note = (
            f"Goalkeeper work says a lot about how hard each side made the other box defend. {hn}'s goalkeeper made {hs} saves and {an}'s made {a_s}. "
            f"That matters most when read beside xGoT, because it shows whether the shots on target were merely counted or genuinely forced intervention from the keeper."
        )
    elif kind == "shared_xg_tiles":
        statline = f"xG {h['xG']}-{a['xG']}  |  xGoT {h['xGoT']}-{a['xGoT']}  |  On target {h['on_target']}-{a['on_target']}"
        note = (
            f"This summary separates chance creation from shot execution. A team can build strong xG and still underperform if the final strike quality drops, which is why xGoT is useful beside the raw shot count. "
            f"Here, the balance between xG and xGoT helps show whether the attacks ended with clean contact or only decent positions."
        )
    elif kind == "shared_match_stats":
        statline = f"Pass accuracy {h['pass_accuracy']}%-{a['pass_accuracy']}%  |  Progressive passes {h['prog_passes']}-{a['prog_passes']}"
        note = (
            f"The headline numbers frame the match before the report moves into team-specific detail. {hn} returned {h['shots']} shots and {h['xG']} xG, while {an} posted {a['shots']} shots and {a['xG']} xG. "
            f"The passing line matters too: the side that pairs cleaner circulation with more progressive passes usually controls how possession turns into territory."
        )
    elif kind == "shared_territorial":
        statline = f"Attacking-third touches {h['touch_att_pct']}%-{a['touch_att_pct']}%  |  Defensive actions {h['defensive_acts']}-{a['defensive_acts']}"
        note = (
            f"Territorial control is about where the match lived, not just who had the ball. {_leader_name(h['touch_att_pct'], a['touch_att_pct'], hn, an)} spent a bigger share of touches high up the pitch, which usually signals longer attacking phases closer to goal. "
            f"When that territorial edge sits next to a heavier defensive workload for the opponent, the game often starts to tilt without needing huge possession gaps."
        )
    elif kind == "shared_touches":
        statline = f"Touches by thirds  |  {hn}: {h['touch_def_pct']} {h['touch_mid_pct']} {h['touch_att_pct']}  |  {an}: {a['touch_def_pct']} {a['touch_mid_pct']} {a['touch_att_pct']}"
        note = (
            f"Touch distribution shows whether possession was useful or mostly safe. {hn} placed {h['touch_att_pct']}% of their touch volume in the attacking third, compared with {an}'s {a['touch_att_pct']}%. "
            f"That split helps distinguish a team that kept the ball near danger from one that circulated deeper before being forced back or wide."
        )
    elif kind == "shared_xt_per_minute":
        hxt = _xt_total(events, hid)
        axt = _xt_total(events, aid)
        statline = f"Total xT {_fmt_num(hxt, 2)}-{_fmt_num(axt, 2)}"
        note = (
            f"The minute-by-minute xT view is a strong way to read momentum. {_leader_name(hxt, axt, hn, an)} produced the higher overall threat return, which means their best attacking spells carried more danger rather than simply more touches. "
            f"Sharp peaks on this chart usually come from short waves of field control, quick combinations or transition attacks, not from slow possession alone."
        )
    elif kind == "shared_def_summary":
        statline = f"Defensive actions {h['defensive_acts']}-{a['defensive_acts']}  |  Tackles {h['tackles']}-{a['tackles']}"
        note = (
            f"The defensive summary helps separate front-foot defending from long periods of containment. {hn} logged {h['defensive_acts']} defensive actions and {an} logged {a['defensive_acts']}. "
            f"The split between tackles, interceptions and recoveries tells you whether each side stepped into duels, read passing lanes, or simply had to reset the shape and clean up second balls."
        )
    elif kind == "shared_dominating_zone":
        statline = f"Touches {h['touches']}-{a['touches']}  |  Zone 14 actions {h['zone14_touches']}-{a['zone14_touches']}"
        note = (
            f"The domination map shows which parts of the pitch each team could actually own. The side with the larger touch base tends to claim more zones, but the most valuable spaces are still the central attacking ones and the half-spaces around the box. "
            f"That is why the zone view is most useful when read next to the attacking-third and Zone 14 numbers."
        )
    elif kind == "team_shot_map":
        avg_xg = round(team['xG'] / team['shots'], 2) if team['shots'] else 0.0
        statline = f"Shots {team['shots']}  |  On target {team['on_target']}  |  xG {team['xG']}"
        note = (
            f"{team_name} finished with {team['shots']} shots, {team['on_target']} of them on target, for {team['xG']} xG. That comes out at roughly {avg_xg:.2f} xG per shot, which helps explain whether the map is built on clean box access or on a larger number of lower-value attempts. "
            f"Against {opp_name}, the shot pattern matters because it shows not only how often {team_name} reached a final action, but also how dangerous those actions really were."
        )
    elif kind == "team_pass_network":
        statline = f"Passes {team['passes_total']}  |  Accuracy {team['pass_accuracy']}%  |  Progressive {team['prog_passes']}"
        note = (
            f"This network is a strong read on how {team_name} organised the ball. They completed {team['passes_total']} passes at {team['pass_accuracy']}% accuracy, with {team['prog_passes']} progressive passes and {team['fwd_pct']}% of all passes played forward. "
            f"That mix tells you whether the shape was keeping the ball for control, or whether it had enough vertical intent to move {opp_name} back and open the next line."
        )
    elif kind == "team_xt_map":
        statline = f"Total xT {_fmt_num(team_xt, 2)}  |  Progressive passes {team['prog_passes']}"
        note = (
            f"The xT map tracks the actions that actually raised the chance of scoring, not just the ones that looked neat in build-up. {team_name} finished with a total xT return of {_fmt_num(team_xt, 2)}. "
            f"If that number sits above {opp_name}'s {_fmt_num(opp_xt, 2)}, it usually means their forward actions were arriving in more dangerous zones and with better timing."
        )
    elif kind == "team_danger_creation":
        statline = f"Shots {team['shots']}  |  Goals {team['goals']}  |  xG {team['xG']}"
        note = (
            f"This view pulls together the actions that directly led to shots or clear danger. {team_name} turned those sequences into {team['shots']} shots and {team['goals']} goals from {team['xG']} xG. "
            f"When the danger events cluster in repeated waves, it usually points to an attack that could re-enter advanced positions instead of relying on isolated moments."
        )
    elif kind == "team_zone14":
        statline = f"Zone 14 and half-space actions {team['zone14_touches']}"
        note = (
            f"{team_name} logged {team['zone14_touches']} actions in Zone 14 and the half-spaces. That matters because those lanes sit between the opponent's midfield and back line and often become the last clean passing window before a shot or a box entry. "
            f"If this number is healthy, the attack was probably finding players between the lines instead of being pushed straight to the touchline."
        )
    elif kind == "team_pass_thirds":
        prof = _pass_third_profile(events, tid)
        statline = f"Own third {prof['def']}  |  Middle third {prof['mid']}  |  Final third {prof['att']}"
        note = (
            f"The third-based pass map shows where {team_name} started their circulation. They played {prof['def']} passes from the first phase, {prof['mid']} from midfield and {prof['att']} in the final third. "
            f"That balance tells you whether the game plan leaned on patient build-up, midfield control, or repeat attacking-phase possession once the ball had already crossed the halfway line."
        )
    elif kind == "team_progressive_passes":
        prof = _progressive_profile(events, tid)
        statline = f"Progressive passes {prof['total']}  |  Origins {prof['def']}/{prof['mid']}/{prof['att']}"
        note = (
            f"{team_name} completed {prof['total']} progressive passes. The origin split of {prof['def']} from the defensive third, {prof['mid']} from midfield and {prof['att']} high up the pitch shows whether progression had to be built patiently or could continue after the team already established territory. "
            f"That is often the difference between one clean wave of attack and sustained pressure."
        )
    elif kind == "team_crosses":
        prof = _cross_profile(events, tid)
        acc = _safe_pct(prof['succ'], prof['total'])
        statline = f"Crosses {prof['total']}  |  Successful {prof['succ']} ({acc}%)"
        note = (
            f"{team_name} attempted {prof['total']} crosses and completed {prof['succ']} of them. The delivery pattern leaned {_dominant_lane(prof['left'], prof['middle'], prof['right'])}, which usually points to the flank where the attack felt it could isolate the defender or free the full-back. "
            f"The success rate matters here because a high crossing volume only helps if it keeps the opposition box under stress."
        )
    elif kind == "team_def_heatmap":
        statline = f"Defensive actions {team['defensive_acts']}  |  Tackles {team['tackles']}  |  Recoveries {team['recoveries']}"
        note = (
            f"The defensive actions map shows where {team_name} had to solve problems without the ball. They logged {team['defensive_acts']} key defensive actions, including {team['tackles']} tackles, {team['interceptions']} interceptions and {team['recoveries']} recoveries. "
            f"If those actions sit higher than {opp_name}'s pressure zones, the team was stepping forward. If they collect closer to their own box, the picture is closer to a protecting block."
        )
    elif kind == "team_average_positions":
        statline = f"Touch split {team['touch_def_pct']} / {team['touch_mid_pct']} / {team['touch_att_pct']}  |  Pass accuracy {team['pass_accuracy']}%"
        note = (
            f"Average positions sketch the team's base shape across the full match. With {team['touch_att_pct']}% of touches in the attacking third and {team['touch_mid_pct']}% through midfield, {team_name} look {'aggressive' if team['touch_att_pct'] > team['touch_def_pct'] else 'more balanced'} in how the structure occupied the pitch. "
            f"The {team['pass_accuracy']}% pass accuracy supports that reading by showing whether the spacing also held up in execution."
        )
    elif kind == "team_box_entries":
        prof = _box_entry_profile(events, tid)
        statline = f"Box entries {prof['total']}  |  By pass {prof['pass']}  |  By carry {prof['carry']}"
        note = (
            f"{team_name} reached the box {prof['total']} times in open play, with {prof['pass']} entries by pass and {prof['carry']} by carry. The strongest route was {_dominant_lane(prof['left'], prof['middle'], prof['right'])}. "
            f"That split helps show whether the side broke lines through combination play or by driving directly at the back line once the attack was already set."
        )
    elif kind == "team_high_turnovers":
        prof = _high_turnover_profile(events, tid)
        statline = f"High turnovers {prof['total']}  |  Led to shots {prof['led_shot']}  |  Led to goals {prof['led_goal']}"
        note = (
            f"{team_name} registered {prof['total']} high turnovers in the main pressing zone. {prof['led_shot']} of those regains led to a shot and {prof['led_goal']} ended in a goal. "
            f"That is the clearest way to judge whether the press merely won territory or actually turned the recovery into immediate attacking value."
        )
    elif kind == "team_pass_target_zones":
        statline = f"Forward pass rate {team['fwd_pct']}%  |  Attacking-third touches {team['touch_att_pct']}%"
        note = (
            f"This target-zone grid shows where {team_name} wanted the ball to arrive after successful passes. With {team['fwd_pct']}% of passes played forward and {team['touch_att_pct']}% of touches in the attacking third, the picture tells you whether the team could push reception points into useful attacking zones or had to settle earlier in the move. "
            f"The busiest cells are often the clearest clue to how the attack tried to pin the opponent back."
        )
    else:
        statline = meta["title"]
        note = (
            "This visual adds another tactical layer to the match report. Read it together with the surrounding charts so the team shape, chance creation and defensive work all sit in the same context."
        )

    return statline, note


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
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Coloured halves
    ax.add_patch(FancyBboxPatch((0.0,  0.0), 0.50, 1.0, boxstyle="square,pad=0", facecolor=C_RED,  alpha=0.92))
    ax.add_patch(FancyBboxPatch((0.50, 0.0), 0.50, 1.0, boxstyle="square,pad=0", facecolor=C_BLUE, alpha=0.92))
    # Dark centre block
    ax.add_patch(FancyBboxPatch((0.34, 0.0), 0.32, 1.0, boxstyle="square,pad=0", facecolor="#111827", alpha=0.97))

    # ── Team names only ─────────────────────────────────────────────
    ax.text(0.17, 0.52, hn,
            ha="center", va="center", color="white",
            fontsize=14, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="#000")])
    ax.text(0.83, 0.52, an,
            ha="center", va="center", color="white",
            fontsize=14, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="#000")])

    # ── Centre: section title (top) + page title (bottom) ──────────
    ax.text(0.50, 0.72, section_title,
            ha="center", va="center", color="#FFD700", fontsize=9, fontweight="bold")
    ax.text(0.50, 0.30, page_title,
            ha="center", va="center", color="white",   fontsize=12, fontweight="bold")

    # ── Sub-bar: competition | venue | page ────────────────────────
    info_ax = fig.add_axes([0.0, 0.915, 1.0, 0.020])
    info_ax.set_xlim(0, 1); info_ax.set_ylim(0, 1); info_ax.axis("off")
    info_ax.set_facecolor("#e8ecf0")
    info_ax.text(0.02, 0.5, info.get("competition", ""), ha="left",   va="center", color="#374151", fontsize=8)
    info_ax.text(0.50, 0.5, info.get("venue", ""),        ha="center", va="center", color="#374151", fontsize=8)
    info_ax.text(0.98, 0.5, f"Page {page_num} of {total_pages}", ha="right", va="center", color="#374151", fontsize=8)


def _draw_pdf_footer(fig, page_num, total_pages, center_text=""):
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.024, 0.024],
                              transform=fig.transFigure, color="#ffffff", lw=0.8, alpha=0.9))
    fig.text(0.03, 0.013, CREDIT_TOOLS, ha="left", va="bottom", color="#ffffff", fontsize=7.5)
    if center_text:
        fig.text(0.50, 0.013, center_text, ha="center", va="bottom", color="#ffffff", fontsize=7.5)
    fig.text(0.97, 0.013, f"{page_num}/{total_pages}", ha="right", va="bottom", color="#ffffff", fontsize=9, fontweight="bold")
def _render_cover_page(pdf, info, stats, events, total_pages):
    hn, an     = info["home_name"], info["away_name"]
    h,  a      = stats["home"],     stats["away"]
    h_sc, a_sc = _parse_scoreline(info, {hn: {"goals": h["goals"]},
                                         an: {"goals": a["goals"]}})

    PDF_BG = "#000000"
    cover  = plt.figure(figsize=(16, 9), facecolor=PDF_BG)
    cover.patch.set_facecolor(PDF_BG)

    bg_ax = cover.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.axis('off')
    bg_ax.set_facecolor(PDF_BG)

    txt_ax = cover.add_axes([0, 0, 1, 1], zorder=2)
    txt_ax.set_xlim(0, 1)
    txt_ax.set_ylim(0, 1)
    txt_ax.axis('off')

    txt_ax.text(0.25, 0.60, hn,
                ha='center', va='center',
                color=C_RED, fontsize=28, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=5, foreground='#000')])

    txt_ax.text(0.50, 0.60, f"{h_sc}  –  {a_sc}",
                ha='center', va='center',
                color='#FFD700', fontsize=52, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=6, foreground='#000')])

    txt_ax.text(0.75, 0.60, an,
                ha='center', va='center',
                color=C_BLUE, fontsize=28, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=5, foreground='#000')])

    txt_ax.text(0.50, 0.44, "Statistical Tactical Analysis",
                ha='center', va='center',
                color='white', fontsize=18, fontweight='bold')

    txt_ax.text(0.50, 0.34, "By Mostafa Saad",
                ha='center', va='center',
                color='white', fontsize=20, fontweight='bold')

    cover.text(0.97, 0.018, f"1/{total_pages}",
               ha='right', va='bottom', color='white', fontsize=10, fontweight='bold',
               transform=cover.transFigure)

    pdf.savefig(cover, facecolor=PDF_BG)
    plt.close(cover)
def _render_section_page(pdf, info, section, summary, page_num, total_pages):
    color = _section_color(section)
    title = _section_title(section, info)

    PDF_BG  = "#000000"   # AMOLED black
    PDF_DIM = "#9ca3af"

    page = plt.figure(figsize=(16, 9), facecolor=PDF_BG)

    bar = page.add_axes([0.0, 0.93, 1.0, 0.07])
    bar.set_xlim(0, 1); bar.set_ylim(0, 1); bar.axis("off")
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.90))
    bar.text(0.50, 0.52, title, ha="center", va="center", color="white", fontsize=16, fontweight="bold")

    page.text(0.50, 0.72, title, ha="center", va="center", color=color, fontsize=28, fontweight="bold")
    page.text(0.50, 0.63, "The pages that follow keep the visual and the tactical explanation together on the same page.",
              ha="center", va="center", color=PDF_DIM, fontsize=11)

    panel = page.add_axes([0.16, 0.26, 0.68, 0.28])
    panel.set_xlim(0, 1); panel.set_ylim(0, 1); panel.axis("off")
    panel.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02",
                                            facecolor="#0d0d0d", edgecolor="#1f2937", lw=1.2, alpha=0.97))
    panel.text(0.05, 0.86, "Section summary", ha="left", va="center", color=color, fontsize=11, fontweight="bold")
    panel.text(0.05, 0.74, _wrap_panel_text(summary, width=92), ha="left", va="top", color="#ffffff", fontsize=10.5)

    _draw_pdf_footer(page, page_num, total_pages, center_text="Created by Mostafa Saad")
    pdf.savefig(page, facecolor=PDF_BG)
    plt.close(page)


def _render_visual_page(pdf, src_fig, info, meta, statline, commentary, page_num, total_pages):
    PDF_BG = "#000000"   # AMOLED black

    page = plt.figure(figsize=(16, 9), facecolor=PDF_BG)
    section_title = _section_title(meta["section"], info)
    _draw_pdf_header(page, info, meta["title"], section_title, page_num, total_pages)

    img = _figure_to_rgba(src_fig)
    img_ax = page.add_axes([0.03, 0.08, 0.62, 0.82])
    img_ax.set_facecolor("#111827")   # the figure itself stays dark
    img_ax.imshow(img)
    img_ax.axis("off")
    for spine in img_ax.spines.values():
        spine.set_visible(False)

    # ── Right panel: light card ─────────────────────────────────────
    panel_ax = page.add_axes([0.68, 0.08, 0.29, 0.82])
    panel_ax.set_xlim(0, 1); panel_ax.set_ylim(0, 1); panel_ax.axis("off")
    panel_ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1,
                                               boxstyle="round,pad=0.02,rounding_size=0.02",
                                               facecolor="#0a0a0a", edgecolor="#1f2937",
                                               lw=1.2, alpha=0.97))
    color = _section_color(meta["section"])
    panel_ax.text(0.06, 0.95, "Tactical note",
                  ha="left", va="top", color=color, fontsize=11, fontweight="bold")
    panel_ax.text(0.06, 0.89, statline,
                  ha="left", va="top", color="#9ca3af", fontsize=9.3)
    panel_ax.plot([0.06, 0.94], [0.84, 0.84], color="#1f2937", lw=1.0)
    panel_ax.text(0.06, 0.81, _wrap_panel_text(commentary, width=44),
                  ha="left", va="top", color="#ffffff", fontsize=10.2)

    _draw_pdf_footer(page, page_num, total_pages)
    pdf.savefig(page, facecolor=PDF_BG)
    plt.close(page)


def build_tactical_pdf(figs, info, events, xg_data, ts):
    """Assemble the final tactical PDF with shared visuals first, then home, then away."""
    hn, an = info["home_name"], info["away_name"]
    stats = _collect_match_stats(info, events, xg_data)

    safe_hn = hn.replace(' ', '_').replace('/', '_')
    safe_an = an.replace(' ', '_').replace('/', '_')
    pdf_path = f"{SAVE_DIR}/tactical_report_{safe_hn}_vs_{safe_an}_{ts}.pdf"

    console.print("\n[bold cyan]  Writing tactical PDF report...[/bold cyan]")
    console.print(f"[bold cyan]  Building PDF: {pdf_path}[/bold cyan]")

    catalog = [m for m in _build_visual_catalog(info) if m["idx"] <= len(figs)]
    section_rank = {"shared": 0, "home": 1, "away": 2}
    ordered_catalog = sorted(catalog, key=lambda item: (section_rank[item["section"]], item["idx"]))

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
                statline, commentary = _visual_tactical_note(meta, info, events, xg_data, stats)
                _render_visual_page(pdf, figs[meta["idx"] - 1], info, meta, statline, commentary, page_num, total_pages)
                page_num += 1

        # TACTICAL SUMMARY PAGE — deleted per user request
        d = pdf.infodict()
        d["Title"] = f"Tactical Report: {hn} {info.get('score', '')} {an}"
        d["Author"] = "Mostafa Saad"
        d["Subject"] = f"{info.get('competition', '')} - {info.get('date', '')}"
        d["Keywords"] = "football tactical report, match analysis, PDF"

    console.print(
        f"\n[bold green]  Tactical PDF saved -> {pdf_path}[/bold green]\n"
        f"  [dim]{len(ordered_catalog)} visual pages grouped into shared, home and away sections[/dim]"
    )
    return pdf_path


# ══════════════════════════════════════════════════════
#  TERMINAL SUMMARY
# ══════════════════════════════════════════════════════
def print_summary(info,xg_data,events):
    console.rule(f"[bold cyan]  {info['home_name']}  {info['score']}  "
                 f"{info['away_name']}  [/bold cyan]")
    console.print(f"  Venue: {info['venue']}   |   "
                  f"Formations: {info['home_form']} vs {info['away_form']}",
                  justify="center")
    xt = Table(title="Shot Breakdown",header_style="bold magenta",
               show_lines=True,border_style="dim")
    for col,style in [
        ("Team","cyan"),("xG","green"),("Shots",""),("On Target","green"),
        ("Goals","yellow"),("Saved","green"),("Missed","red"),
        ("Blocked","orange3"),("Post","blue"),("Big Ch.",""),
    ]:
        xt.add_column(col,style=style,justify="center",
                      min_width=16 if col=="Team" else 7)
    for name,s in xg_data.items():
        xt.add_row(name,str(s["xG"]),str(s["shots"]),str(s["on_target"]),
                   str(s["goals"]),str(s["saved"]),str(s["missed"]),
                   str(s["blocked"]),str(s["post"]),str(s["big_chances"]))
    console.print(xt)

    pss = events[events["is_pass"]==True]
    if not pss.empty:
        pt = Table(title="Pass Stats",header_style="bold blue",
                   show_lines=True,border_style="dim")
        for col,style,just in [
            ("Team","cyan","left"),("Total","","center"),
            ("Completed","green","center"),("Accuracy","green","center"),
            ("Key Passes","yellow","center"),
        ]:
            pt.add_column(col,style=style,justify=just,
                          min_width=16 if col=="Team" else 8)
        for side in ["home","away"]:
            tid=info[f"{side}_id"]; name=info[f"{side}_name"]
            tp=pss[pss["team_id"]==tid]; tot=len(tp)
            suc=int((tp["outcome"]=="Successful").sum())
            acc=round(suc/tot*100,1) if tot else 0
            key=int(tp["is_key_pass"].sum())
            pt.add_row(name,str(tot),str(suc),f"{acc}%",str(key))
        console.print(pt)

    gdf = events[events["is_goal"]==True]
    if not gdf.empty:
        gt = Table(title="Goals",header_style="bold yellow",
                   show_lines=True,border_style="dim")
        gt.add_column("Min",       justify="center",width=5)
        gt.add_column("Scorer",    style="bold white",min_width=18)
        gt.add_column("Scored For",style="cyan",min_width=14)
        gt.add_column("Type",      justify="center",width=12)
        gt.add_column("Assist",    style="green",min_width=18)
        gt.add_column("xG",        justify="center",style="yellow",width=6)
        for _,row in gdf.iterrows():
            scored_for = info["home_name"] \
                         if row["scoring_team"]==info["home_id"] \
                         else info["away_name"]
            goal_type  = "[bold magenta]🔄 OWN GOAL[/bold magenta]" \
                         if row.get("is_own_goal",False) \
                         else ("🟡 Penalty" if row["is_penalty"]
                         else ("🔵 Header"  if row["is_header"]
                         else "⚽ Open Play"))
            gt.add_row(f"{row['minute']}'",_short(str(row["player"])),
                       scored_for,goal_type,
                       _short(str(row["assist_player"])) \
                           if row["assist_player"] else "—",
                       f"{row['xG']:.3f}" if row["xG"] else "—")
        console.print(gt)


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Scrape بـ 3 محاولات تلقائية ─────────────────
    md = scrape_match(
        MATCH_URL,
        chromedriver_path=CHROMEDRIVER_PATH,
        profile_dir=CHROME_PROFILE_DIR,
        profile_name=CHROME_PROFILE_NAME,
    )

    info, events, players = parse_all(md)
    xg_data   = xg_stats(events, info)
    status    = get_status(md)
    sub_in    = info["sub_in"]
    sub_out   = info["sub_out"]
    red_cards = info["red_cards"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    events.to_csv( f"{SAVE_DIR}/events_{ts}.csv",  index=False, encoding="utf-8-sig")
    players.to_csv(f"{SAVE_DIR}/players_{ts}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(xg_data).T.reset_index().rename(
        columns={"index":"team"}).to_csv(
        f"{SAVE_DIR}/xg_{ts}.csv", index=False, encoding="utf-8-sig")

    print_summary(info, xg_data, events)

    plt.style.use("dark_background")
    figs = []

    # ── shorthand — available to ALL helpers and figure calls below ───
    hn,  an  = info["home_name"], info["away_name"]
    hid, aid = info["home_id"],   info["away_id"]

    def _fig(w,h,title=""):
        f = plt.figure(figsize=(w,h), facecolor=BG_DARK)
        if title: f.canvas.manager.set_window_title(title)
        return f

    # ── shared header helpers (must be defined before any fig call) ────────
    def _add_header(fig, title, subtitle=""):
        """
        Colour bar (home | away), stat name, key numbers.
        Credit "Created by Mostafa Saad" in CENTRE of the colour bar.
        """
        # ── colour bar ────────────────────────────────────────────
        cax = fig.add_axes([0.0, 0.981, 1.0, 0.019])
        cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.axis("off")
        cax.add_patch(plt.Rectangle((0,    0), 0.50, 1,
                                    facecolor=C_RED,  alpha=0.92, zorder=0))
        cax.add_patch(plt.Rectangle((0.50, 0), 0.50, 1,
                                    facecolor=C_BLUE, alpha=0.92, zorder=0))
        cax.plot([0.50, 0.50], [0.08, 0.92],
                 color="white", lw=0.8, alpha=0.35, zorder=2)
        # Home name — left
        cax.text(0.015, 0.50, f"● {hn[:20]}", ha="left", va="center",
                 color="white", fontsize=7.5, fontweight="bold", zorder=3)
        # Credit — CENTRE of bar
        cax.text(0.50, 0.50, "Created by Mostafa Saad",
                 ha="center", va="center", color="#FFD700",
                 fontsize=7.8, fontweight="bold", fontstyle="italic", zorder=3)
        # Away name — right
        cax.text(0.985, 0.50, f"{an[:20]} ●", ha="right", va="center",
                 color="white", fontsize=7.5, fontweight="bold", zorder=3)
        # ── Line 1: stat name (with glow) ─────────────────────────
        fig.text(0.50, 0.966, title,
                 ha="center", va="top",
                 color=TEXT_BRIGHT, fontsize=15, fontweight="bold",
                 transform=fig.transFigure,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
        # ── Line 2: key numbers ───────────────────────────────────
        if subtitle:
            fig.text(0.50, 0.928, subtitle,
                     ha="center", va="top",
                     color=TEXT_DIM, fontsize=9,
                     transform=fig.transFigure)

    # ── 1. xG Flow ───────────────────────────────────
    fig1 = _fig(15,7,"Fig 1 — xG Flow")
    ax1  = fig1.add_subplot(
        GridSpec(1,1,figure=fig1,left=0.07,right=0.97,top=0.88,bottom=0.11)[0,0])
    _add_header(fig1, "xG Flow",
        f"{hn}: xG {xg_data.get(hn,{}).get('xG',0):.2f}  |  {an}: xG {xg_data.get(an,{}).get('xG',0):.2f}")
    draw_xg_flow(fig1,ax1,events,info,xg_data,status)
    _watermark(fig1)
    fig1.savefig(f"{SAVE_DIR}/1_xg_flow_{ts}.png",
                 dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig1)

    # ── 2. Shot Map Home ─────────────────────────────
    fig2 = _fig(14,12,f"Fig 2 — Shot Map: {info['home_name']}")
    draw_shot_map_full(fig2,events,info["home_id"],info["home_name"],C_RED)
    _watermark(fig2)
    fig2.savefig(f"{SAVE_DIR}/2_shot_map_home_{ts}.png",
                 dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig2)

    # ── 3. Shot Map Away ─────────────────────────────
    fig3 = _fig(14,12,f"Fig 3 — Shot Map: {info['away_name']}")
    draw_shot_map_full(fig3,events,info["away_id"],info["away_name"],C_BLUE)
    _watermark(fig3)
    fig3.savefig(f"{SAVE_DIR}/3_shot_map_away_{ts}.png",
                 dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig3)

    # ── 4. Breakdown + Goals ─────────────────────────
    fig4 = _fig(16,13,"Fig 4 — Breakdown & Goals")
    _add_header(fig4, "Shot Breakdown & Goals",
        f"{hn}: {xg_data.get(hn,{}).get('shots',0)} shots  xG {xg_data.get(hn,{}).get('xG',0):.2f}"
        f"   |   {an}: {xg_data.get(an,{}).get('shots',0)} shots  xG {xg_data.get(an,{}).get('xG',0):.2f}")
    draw_breakdown_goals(fig4,events,info,xg_data)
    _watermark(fig4)
    fig4.savefig(f"{SAVE_DIR}/4_breakdown_goals_{ts}.png",
                 dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig4)

    # ── 5. Pass Network Home ─────────────────────────
    fig5 = _fig(18,11,f"Fig 5 — Pass Network: {info['home_name']}")
    draw_pass_network_full(fig5,events,info["home_id"],info["home_name"],
                           C_RED,sub_in,sub_out,red_cards)
    _watermark(fig5)
    fig5.savefig(f"{SAVE_DIR}/5_pass_network_home_{ts}.png",
                 dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig5)

    # ── 6. Pass Network Away ─────────────────────────
    fig6 = _fig(18,11,f"Fig 6 — Pass Network: {info['away_name']}")
    draw_pass_network_full(fig6,events,info["away_id"],info["away_name"],
                           C_BLUE,sub_in,sub_out,red_cards)
    _watermark(fig6)
    fig6.savefig(f"{SAVE_DIR}/6_pass_network_away_{ts}.png",
                 dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig6)

    # ── 7. xT Map Home ───────────────────────────────
    fig7 = _fig(18,11,f"Fig 7 — xT Map: {info['home_name']}")
    draw_xt_map_full(fig7,events,info["home_id"],info["home_name"],C_RED)
    _watermark(fig7)
    fig7.savefig(f"{SAVE_DIR}/7_xt_map_home_{ts}.png",
                 dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig7)

    # ── 8. xT Map Away ──────────────────────────────
    fig8 = _fig(18,11,f"Fig 8 — xT Map: {info['away_name']}")
    draw_xt_map_full(fig8,events,info["away_id"],info["away_name"],C_BLUE)
    _watermark(fig8)
    fig8.savefig(f"{SAVE_DIR}/8_xt_map_away_{ts}.png",
                  dpi=150,bbox_inches="tight",facecolor=BG_DARK)
    figs.append(fig8)

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
        cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.axis("off")

        if team_color and team_name:
            # Single team: full-width band in team colour
            cax.add_patch(plt.Rectangle((0, 0), 1.0, 1,
                                        facecolor=team_color, alpha=0.93, zorder=0))
            # Subtle highlight strip at top edge
            cax.add_patch(plt.Rectangle((0, 0.82), 1.0, 0.18,
                                        facecolor="white", alpha=0.07, zorder=1))
            # Team name — left
            cax.text(0.015, 0.50, f"● {team_name}",
                     ha="left", va="center", color="white",
                     fontsize=8.5, fontweight="bold", zorder=3)
            # Credit — centre
            cax.text(0.50, 0.50, "Created by Mostafa Saad",
                     ha="center", va="center", color="#FFD700",
                     fontsize=8, fontweight="bold", fontstyle="italic", zorder=3)
        else:
            # Both teams: split band
            cax.add_patch(plt.Rectangle((0, 0),    0.50, 1,
                                        facecolor=C_RED,  alpha=0.91, zorder=0))
            cax.add_patch(plt.Rectangle((0.50, 0), 0.50, 1,
                                        facecolor=C_BLUE, alpha=0.91, zorder=0))
            # Thin white separator at centre
            cax.plot([0.50, 0.50], [0.08, 0.92],
                     color="white", lw=0.8, alpha=0.35, zorder=2)
            # Home name — left
            cax.text(0.015, 0.50, f"● {hn[:18]}",
                     ha="left", va="center", color="white",
                     fontsize=8, fontweight="bold", zorder=3)
            # Credit — centre
            cax.text(0.50, 0.50, "Created by Mostafa Saad",
                     ha="center", va="center", color="#FFD700",
                     fontsize=7.8, fontweight="bold", fontstyle="italic", zorder=3)
            # Away name — right
            cax.text(0.985, 0.50, f"{an[:18]} ●",
                     ha="right", va="center", color="white",
                     fontsize=8, fontweight="bold", zorder=3)

        # ── Line 1: stat name (with glow) ─────────────────────────────
        f.text(0.50, 0.962, label,
               ha="center", va="top",
               color=TEXT_BRIGHT, fontsize=15, fontweight="bold",
               transform=f.transFigure,
               path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])

        # ── Line 2: key numbers ───────────────────────────────────────
        if subtitle:
            f.text(0.50, 0.928, subtitle,
                   ha="center", va="top",
                   color=TEXT_DIM, fontsize=9,
                   transform=f.transFigure)

        return f

    def _sp(fig, lp=0.06, rp=0.95, tp=0.82, bp=0.10):
        """Single subplot below the three-line header (title + subtitle + credit).
        top=0.82  → clears header area
        bottom=0.10 → room for axis labels + watermark
        """
        return fig.add_subplot(
            GridSpec(1,1,figure=fig,left=lp,right=rp,top=tp,bottom=bp)[0,0])

    def _sv(fig, fname):
        """Watermark + save."""
        _watermark(fig)
        fig.savefig(fname, dpi=100, bbox_inches="tight", facecolor=BG_DARK)
        figs.append(fig)

    base = f"{SAVE_DIR}"

    # ── pre-compute subtitle data ─────────────────────────────────
    _hxg   = xg_data.get(hn, {})
    _axg   = xg_data.get(an, {})
    _h_xg  = _hxg.get("xG",   0);  _a_xg  = _axg.get("xG",   0)
    _h_xgt = _hxg.get("xGoT", 0);  _a_xgt = _axg.get("xGoT", 0)
    _h_sh  = _hxg.get("shots",0);  _a_sh  = _axg.get("shots",0)
    _h_ot  = _hxg.get("on_target",0); _a_ot = _axg.get("on_target",0)
    _h_sv  = _hxg.get("saved", 0); _a_sv  = _axg.get("saved", 0)

    # ── A: Shot Comparison tiles (both) ──────────────────────────
    fa = _sf(13, 5.0, "Shot Comparison",
             subtitle=f"{hn}  vs  {an}   |   xG: {_h_xg:.2f} – {_a_xg:.2f}   |   Shots: {_h_sh} – {_a_sh}")
    _panel_shot_comparison(_sp(fa), events, info, xg_data)
    _sv(fa, f"{base}/11_shot_comparison_{ts}.png")

    # ── B: Danger Creation — Home ─────────────────────────────────
    fb = _sf(10, 8, "Danger Creation", team_color=C_RED, team_name=hn,
             subtitle=f"{hn}   |   Shots: {_h_sh}   On Target: {_h_ot}   xG: {_h_xg:.2f}")
    _panel_danger(_sp(fb, lp=0.02, rp=0.98), events, hid, C_RED, hn)
    _sv(fb, f"{base}/12_danger_home_{ts}.png")

    # ── C: Danger Creation — Away ─────────────────────────────────
    fc = _sf(10, 8, "Danger Creation", team_color=C_BLUE, team_name=an,
             subtitle=f"{an}   |   Shots: {_a_sh}   On Target: {_a_ot}   xG: {_a_xg:.2f}")
    _panel_danger(_sp(fc, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
    _sv(fc, f"{base}/13_danger_away_{ts}.png")

    # ── D: GK Saves (both) ────────────────────────────────────────
    fd = _sf(10, 6, "Goalkeeper Saves",
             subtitle=f"{hn}: {_h_sv} saves   |   {an}: {_a_sv} saves")
    _panel_gk_saves(_sp(fd), events, info)
    _sv(fd, f"{base}/14_gk_saves_{ts}.png")

    # ── E: xG / xGoT / OnTarget tiles (both) ─────────────────────
    fe = _sf(13, 5.0, "xG / xGoT / On Target",
             subtitle=f"{hn}  xG {_h_xg:.2f}   xGoT {_h_xgt:.2f}   |   {an}  xG {_a_xg:.2f}   xGoT {_a_xgt:.2f}")
    _panel_xg_tiles(_sp(fe), events, info, xg_data)
    _sv(fe, f"{base}/15_xg_tiles_{ts}.png")

    # ── F: Zone 14 & Half-Spaces — Home ──────────────────────────
    ff = _sf(10, 8, "Zone 14 & Half-Spaces", team_color=C_RED, team_name=hn,
         subtitle=f"{hn}   |   Actions in Zone 14 and Half-Space channels")
    _panel_zone14(_sp(ff, lp=0.02, rp=0.98), events, hid, C_RED, hn)
    _sv(ff, f"{base}/16_zone14_home_{ts}.png")

    # ── G: Zone 14 & Half-Spaces — Away ──────────────────────────
    fg = _sf(10, 8, "Zone 14 & Half-Spaces", team_color=C_BLUE, team_name=an,
         subtitle=f"{an}   |   Actions in Zone 14 and Half-Space channels")
    _panel_zone14(_sp(fg, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
    _sv(fg, f"{base}/17_zone14_away_{ts}.png")

    # ── H: Match Statistics (both) ───────────────────────────────
    fh = _sf(9, 10, "Match Statistics",
         subtitle=f"{hn}  vs  {an}   |   {info['venue']}")
    _panel_match_stats(_sp(fh, lp=0.07, rp=0.93), events, info, xg_data)
    _sv(fh, f"{base}/18_match_stats_{ts}.png")

    # ── I: Territorial Control (both) ────────────────────────────
    fi = _sf(9, 6, "Territorial Control",
         subtitle=f"{hn}  vs  {an}   |   Events per pitch third")
    _panel_territorial(_sp(fi, lp=0.18, rp=0.96), events, info)
    _sv(fi, f"{base}/19_territorial_{ts}.png")

    # ── J: Possession / Ball Touches (both) ──────────────────────
    fj = _sf(9, 6, "Ball Touches",
         subtitle=f"{hn}  vs  {an}   |   Touch distribution by zone")
    _panel_donut_dual(_sp(fj), events, info)
    _sv(fj, f"{base}/20_possession_{ts}.png")

    # ── K: Pass Map / Thirds — Home ──────────────────────────────
    fk = _sf(10, 8, "Pass Map by Third", team_color=C_RED, team_name=hn,
         subtitle=f"{hn}   |   Completed and incomplete passes across thirds")
    _panel_pass_thirds(_sp(fk, lp=0.02, rp=0.98), events, hid, C_RED, hn)
    _sv(fk, f"{base}/21_pass_thirds_home_{ts}.png")

    # ── L: Pass Map / Thirds — Away ──────────────────────────────
    fl = _sf(10, 8, "Pass Map by Third", team_color=C_BLUE, team_name=an,
         subtitle=f"{an}   |   Completed and incomplete passes across thirds")
    _panel_pass_thirds(_sp(fl, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
    _sv(fl, f"{base}/22_pass_thirds_away_{ts}.png")

    # ── M: xT per Minute (both) ──────────────────────────────────
    fm = _sf(12, 6, "xT per Minute",
         subtitle=f"{hn} (▲ red)  vs  {an} (▼ blue)   |   Expected Threat generated each minute")
    _panel_xt_minute(_sp(fm, lp=0.08, rp=0.97, bp=0.11), events, info)
    _sv(fm, f"{base}/23_xt_per_minute_{ts}.png")

    # ── N: Progressive Passes — Home ─────────────────────────────
    fn_ = _sf(10, 8, "Progressive Passes", team_color=C_RED, team_name=hn,
          subtitle=f"{hn}   |   Passes moving the ball significantly toward goal")
    _panel_progressive(_sp(fn_, lp=0.02, rp=0.98), events, hid, C_RED, hn)
    _sv(fn_, f"{base}/24_progressive_home_{ts}.png")

    # ── O: Progressive Passes — Away ─────────────────────────────
    fo = _sf(10, 8, "Progressive Passes", team_color=C_BLUE, team_name=an,
         subtitle=f"{an}   |   Passes moving the ball significantly toward goal")
    _panel_progressive(_sp(fo, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
    _sv(fo, f"{base}/25_progressive_away_{ts}.png")

    # ── P1: Crosses — Home ───────────────────────────────────────
    fp1 = _sf(10, 8, "Crosses", team_color=C_RED, team_name=hn,
          subtitle=f"{hn}   |   Successful (solid) and unsuccessful (faded) crosses")
    _panel_crosses_team(_sp(fp1, lp=0.02, rp=0.98), events, hid, C_RED, hn)
    _sv(fp1, f"{base}/26_crosses_home_{ts}.png")

    # ── P2: Crosses — Away ───────────────────────────────────────
    fp2 = _sf(10, 8, "Crosses", team_color=C_BLUE, team_name=an,
          subtitle=f"{an}   |   Successful (solid) and unsuccessful (faded) crosses")
    _panel_crosses_team(_sp(fp2, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
    _sv(fp2, f"{base}/27_crosses_away_{ts}.png")

    # ── Q: Defensive Heatmap — Home ──────────────────────────────
    fq = _sf(10, 8, "Defensive Actions", team_color=C_RED, team_name=hn,
         subtitle=f"{hn}   |   Tackles · Interceptions · Recoveries · Clearances · Aerials")
    _panel_defensive_heatmap(_sp(fq, lp=0.02, rp=0.98), events, hid, C_RED, hn)
    _sv(fq, f"{base}/28_defensive_hm_home_{ts}.png")

    # ── R: Defensive Heatmap — Away ──────────────────────────────
    fr = _sf(10, 8, "Defensive Actions", team_color=C_BLUE, team_name=an,
         subtitle=f"{an}   |   Tackles · Interceptions · Recoveries · Clearances · Aerials")
    _panel_defensive_heatmap(_sp(fr, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
    _sv(fr, f"{base}/29_defensive_hm_away_{ts}.png")

    # ── S: Defensive Summary Table (both) ────────────────────────
    fs = _sf(9, 8, "Defensive Summary",
         subtitle=f"{hn}  vs  {an}   |   Count of each defensive action type")
    _panel_def_counts(_sp(fs, lp=0.07, rp=0.93), events, info)
    _sv(fs, f"{base}/30_defensive_summary_{ts}.png")

    # ── T: Avg Positions — Home ───────────────────────────────────
    ft = _sf(10, 8, "Average Positions", team_color=C_RED, team_name=hn,
         subtitle=f"{hn}   |   Mean touch position per player (size = touches)")
    _panel_avg_position(_sp(ft, lp=0.02, rp=0.98), events, hid, C_RED, hn)
    _sv(ft, f"{base}/31_avg_position_home_{ts}.png")

    # ── U: Avg Positions — Away ───────────────────────────────────
    fu = _sf(10, 8, "Average Positions", team_color=C_BLUE, team_name=an,
         subtitle=f"{an}   |   Mean touch position per player (size = touches)")
    _panel_avg_position(_sp(fu, lp=0.02, rp=0.98), events, aid, C_BLUE, an)
    _sv(fu, f"{base}/32_avg_position_away_{ts}.png")

    # ══════════════════════════════════════════════════════
    #  NEW FIGURES 33–41
    # ══════════════════════════════════════════════════════

    # ── 33: Dominating Zone (both) ────────────────────────
    f33 = _sf(14, 8, "Dominating Zone",
              subtitle=f"{hn}  vs  {an}  |  >55% touches = dominant  |  45-55% = contested")
    _panel_dominating_zone(_sp(f33, lp=0.03, rp=0.97, tp=0.84, bp=0.10),
                           events, info)
    _sv(f33, f"{base}/33_dominating_zone_{ts}.png")

    # ── 34: Box Entries — Home ────────────────────────────
    f34 = _sf(8, 11, "Box Entries", team_color=C_RED, team_name=hn,
              subtitle=f"{hn}  |  Passes & carries ending in opponent's penalty box")
    _panel_box_entries(_sp(f34, lp=0.05, rp=0.95, tp=0.84, bp=0.06),
                       events, hid, C_RED, hn)
    _sv(f34, f"{base}/34_box_entries_home_{ts}.png")

    # ── 35: Box Entries — Away ────────────────────────────
    f35 = _sf(8, 11, "Box Entries", team_color=C_BLUE, team_name=an,
              subtitle=f"{an}  |  Passes & carries ending in opponent's penalty box")
    _panel_box_entries(_sp(f35, lp=0.05, rp=0.95, tp=0.84, bp=0.06),
                       events, aid, C_BLUE, an)
    _sv(f35, f"{base}/35_box_entries_away_{ts}.png")

    # ── 36: High Turnovers — Home ────────────────────────
    f36 = _sf(8, 11, "High Turnovers", team_color=C_RED, team_name=hn,
              subtitle=f"{hn}  |  Ball wins within 40m of opponent goal")
    _panel_high_turnovers(_sp(f36, lp=0.05, rp=0.95, tp=0.84, bp=0.06),
                          events, hid, C_RED, hn)
    _sv(f36, f"{base}/36_high_turnovers_home_{ts}.png")

    # ── 37: High Turnovers — Away ────────────────────────
    f37 = _sf(8, 11, "High Turnovers", team_color=C_BLUE, team_name=an,
              subtitle=f"{an}  |  Ball wins within 40m of opponent goal")
    _panel_high_turnovers(_sp(f37, lp=0.05, rp=0.95, tp=0.84, bp=0.06),
                          events, aid, C_BLUE, an)
    _sv(f37, f"{base}/37_high_turnovers_away_{ts}.png")

    # ── 38: Pass Target Zones — Home ─────────────────────
    f38 = _sf(8, 11, "Pass Target Zones", team_color=C_RED, team_name=hn,
              subtitle=f"{hn}  |  % of successful passes received per zone")
    _panel_pass_target_zones(_sp(f38, lp=0.05, rp=0.95, tp=0.84, bp=0.06),
                             events, hid, C_RED, hn)
    _sv(f38, f"{base}/38_pass_target_home_{ts}.png")

    # ── 39: Pass Target Zones — Away ─────────────────────
    f39 = _sf(8, 11, "Pass Target Zones", team_color=C_BLUE, team_name=an,
              subtitle=f"{an}  |  % of successful passes received per zone")
    _panel_pass_target_zones(_sp(f39, lp=0.05, rp=0.95, tp=0.84, bp=0.06),
                             events, aid, C_BLUE, an)
    _sv(f39, f"{base}/39_pass_target_away_{ts}.png")

    # ══════════════════════════════════════════════════════
    #  TACTICAL PDF REPORT
    # ══════════════════════════════════════════════════════
    try:
        build_tactical_pdf(figs, info, events, xg_data, ts)
    except Exception as _pdf_err:
        console.print(f"[yellow]  ⚠ PDF generation failed: {_pdf_err}[/yellow]")
        import traceback; traceback.print_exc()

    total_figs = 8 + 22 + 7   # 39 total
    console.print(
        f"\n[bold green]  ✅ {total_figs} figures saved → {SAVE_DIR}/[/bold green]\n"
        f"  [dim]Figs  1-8  : individual analytics[/dim]\n"
        f"  [dim]Figs  9-32 : standalone visuals[/dim]\n"
        f"  [dim]Figs 33-39 : Dominating Zone · Box Entries · High Turnovers · Pass Target Zones[/dim]"
    )
    plt.show()


if __name__ == "__main__":
    main()
