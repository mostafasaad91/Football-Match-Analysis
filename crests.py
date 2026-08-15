"""Club crests for the match posters.

The provider's own team id is already carried on every event row, and the same
id addresses the provider's crest CDN, so a crest needs no name matching and no
lookup table -- the mapping that usually breaks the moment a club is written
"Wolves" in one source and "Wolverhampton Wanderers" in another.

Crests are cached under ``assets/crests`` and that directory is not tracked:
club crests are trademarks, fine to render on an analysis poster with
attribution, not ours to redistribute in a repository.

Resolution order for one team:

1. ``assets/crests/<team_id>.png`` -- a local override, for dropping in a
   higher-resolution crest than the CDN serves.
2. the cached download of the CDN copy.
3. a monogram roundel drawn in the side's kit colour, so a poster never fails
   to build because a crest could not be fetched.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "assets" / "crests"
CREST_URL = "https://d2zywfiolv4f83.cloudfront.net/img/teams/{team_id}.png"
_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0"

# The CDN serves 70x70. Anything drawn much larger than this goes soft, so the
# posters place crests near their native size and the local override exists for
# anyone who wants them bigger.
NATIVE_PX = 70

# Below this contrast against the page, a crest is laid on a light plate rather
# than left to disappear -- the same floor the marks use.
_PLATE_CONTRAST_FLOOR = 2.6

_MEMO: dict[int, np.ndarray | None] = {}


def _relative_luminance(rgb) -> float:
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(rgb, background) -> float:
    bright, dark = sorted((_relative_luminance(rgb), _relative_luminance(background)))
    return (dark + 0.05) / (bright + 0.05)


def cache_path(team_id: int) -> Path:
    return CACHE_DIR / f"{int(team_id)}.png"


def _load(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None


def download_crest(team_id: int, *, force: bool = False) -> Path | None:
    """Fetch one crest into the cache and return its path.

    Returns ``None`` rather than raising when the crest cannot be fetched: a
    missing crest costs a poster its badge, not its build.
    """
    target = cache_path(team_id)
    if target.exists() and not force:
        return target
    request = urllib.request.Request(
        CREST_URL.format(team_id=int(team_id)), headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            payload = response.read()
        # Decode before writing, so a captive-portal HTML page or an error body
        # never lands in the cache under a .png name and poisons later runs.
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def crest_image(team_id: int, *, allow_download: bool = True) -> np.ndarray | None:
    """Return one team's crest as RGBA, or ``None`` if there isn't one."""
    key = int(team_id)
    if key in _MEMO:
        return _MEMO[key]
    path = cache_path(key)
    image = _load(path) if path.exists() else None
    if image is None and allow_download and download_crest(key) is not None:
        image = _load(path)
    _MEMO[key] = image
    return image


def needs_plate(image: np.ndarray, background: str) -> bool:
    """Whether a crest is too dark for the page it is being drawn on.

    Averaged over the opaque pixels only -- a crest is mostly transparent
    corner, and including those would call every crest light enough.
    """
    alpha = image[..., 3].astype(float) / 255.0
    weight = alpha.sum()
    if weight <= 0:
        return False
    rgb = [float((image[..., channel].astype(float) / 255.0 * alpha).sum() / weight)
           for channel in range(3)]
    return _contrast(rgb, mcolors.to_rgb(background)) < _PLATE_CONTRAST_FLOOR


def place_crest(
    fig,
    x: float,
    y: float,
    team_id: int,
    *,
    monogram: str,
    colour: str,
    size_px: float,
    background: str = "#000000",
    text_colour: str = "#FFFFFF",
    allow_download: bool = True,
) -> bool:
    """Draw one club's crest at a figure-fraction position.

    Returns True when a real crest was drawn, False when the monogram roundel
    stood in for it. ``size_px`` is the crest's width in output pixels.
    """
    image = crest_image(team_id, allow_download=allow_download)
    width_frac = size_px / fig.get_figwidth() / fig.dpi
    height_frac = size_px / fig.get_figheight() / fig.dpi

    if image is None:
        # Monogram roundel: an ellipse in figure fractions, since the figure is
        # not square and a circle in those units would not be round.
        fig.add_artist(Ellipse((x, y), width_frac, height_frac, facecolor=colour,
                               edgecolor="none", zorder=6))
        fig.add_artist(Ellipse((x, y), width_frac * 0.78, height_frac * 0.78,
                               facecolor="none", edgecolor=text_colour, lw=1.1,
                               alpha=0.5, zorder=7))
        fig.text(x, y, monogram, color=text_colour, fontsize=size_px * 0.115,
                 fontweight="bold", ha="center", va="center", zorder=8)
        return False

    if needs_plate(image, background):
        pad_w, pad_h = width_frac * 0.14, height_frac * 0.14
        fig.add_artist(FancyBboxPatch(
            (x - width_frac / 2 - pad_w, y - height_frac / 2 - pad_h),
            width_frac + 2 * pad_w, height_frac + 2 * pad_h,
            boxstyle="round,pad=0,rounding_size=0.008",
            facecolor="#E8EBEF", edgecolor="none", alpha=0.93, zorder=5,
        ))

    # OffsetImage zoom is in 72dpi points, so the figure's dpi has to be
    # divided out or the crest lands dpi/72 times too large.
    zoom = size_px / image.shape[1] / (fig.dpi / 72.0)
    fig.add_artist(AnnotationBbox(
        OffsetImage(image, zoom=zoom, interpolation="lanczos"),
        (x, y), xycoords="figure fraction", frameon=False, zorder=6,
    ))
    return True


def prefetch(team_ids) -> dict[int, bool]:
    """Warm the cache for a fixture. Returns which ids resolved to a crest."""
    return {int(tid): crest_image(int(tid)) is not None for tid in team_ids}
