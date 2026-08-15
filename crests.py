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


def _draw(fig, image: np.ndarray, x: float, y: float, width: float,
          zorder: float) -> None:
    """Paint ``image`` centred on a figure-fraction point, at ``width`` of it.

    Drawn into its own transparent axes rather than as an AnnotationBbox. The
    offsetbox resolved its figure-fraction anchor against the cropped box when
    the caller saved with ``bbox_inches="tight"``, which put every crest in the
    visuals' header up and to the right of where it was placed, clipped by the
    page edge. An axes rectangle is in figure coordinates by definition and is
    unaffected by the crop. It also removes the dpi arithmetic: the earlier
    zoom divided by the figure's dpi and the visuals save at a different one,
    so each crest also came out half again too large.
    """
    height = width * fig.get_figwidth() / fig.get_figheight()
    ax = fig.add_axes([x - width / 2, y - height / 2, width, height],
                      zorder=zorder)
    ax.imshow(image, interpolation="lanczos", aspect="auto")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)

# A crest earns a plate when too little of it separates from the page. Judged
# per pixel rather than on the crest's mean colour: Aston Villa's averages
# light, because of the pale shield behind the lion, but its claret border and
# blue field both clear the page comfortably and it needs no plate at all.
_PIXEL_CONTRAST_FLOOR = 2.0
_MIN_READABLE_FRACTION = 0.30

# The plate itself. Slate rather than near-black on paper: a crest tile is
# meant to read as a deliberate badge, not as a hole in the page.
PLATE_ON_DARK_PAGE = "#E8EBEF"
PLATE_ON_LIGHT_PAGE = "#39414D"

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


def crest_luminance(image: np.ndarray) -> float:
    """Mean luminance over the opaque pixels of a crest.

    Averaged over the opaque pixels only -- a crest is mostly transparent
    corner, and including those would call every crest light enough.
    """
    alpha = image[..., 3].astype(float) / 255.0
    weight = alpha.sum()
    if weight <= 0:
        return 0.0
    rgb = [float((image[..., channel].astype(float) / 255.0 * alpha).sum() / weight)
           for channel in range(3)]
    return _relative_luminance(rgb)


def readable_fraction(image: np.ndarray, background: str) -> float:
    """Share of a crest's opaque area that separates from the page."""
    alpha = image[..., 3].astype(float) / 255.0
    opaque = alpha > 0.5
    if not opaque.any():
        return 1.0
    rgb = image[..., :3].astype(float) / 255.0
    channels = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = (0.2126 * channels[..., 0] + 0.7152 * channels[..., 1]
                 + 0.0722 * channels[..., 2])
    page = _relative_luminance(mcolors.to_rgb(background))
    bright = np.maximum(luminance, page)
    dark = np.minimum(luminance, page)
    contrast = (bright + 0.05) / (dark + 0.05)
    return float((contrast[opaque] >= _PIXEL_CONTRAST_FLOOR).mean())


def needs_plate(image: np.ndarray, background: str) -> bool:
    """Whether too little of a crest separates from the page under it."""
    return readable_fraction(image, background) < _MIN_READABLE_FRACTION


def plate_colour(image: np.ndarray, background: str) -> str:
    """The plate a crest is laid on when it cannot hold its own on the page.

    A crest reaches this point because most of it sits close to the page, so
    the plate has to move away from the page: light behind a navy crest on
    black, dark behind a white-and-silver one on paper.
    """
    del image  # kept in the signature: the plate is per-crest by intent
    light_page = _relative_luminance(mcolors.to_rgb(background)) > 0.5
    return PLATE_ON_LIGHT_PAGE if light_page else PLATE_ON_DARK_PAGE


def place_crest(
    fig,
    x: float,
    y: float,
    team_id: int,
    *,
    monogram: str,
    colour: str,
    width: float,
    background: str = "#000000",
    text_colour: str | None = None,
    allow_download: bool = True,
    zorder: float = 6.0,
) -> bool:
    """Draw one club's crest at a figure-fraction position.

    Returns True when a real crest was drawn, False when the monogram roundel
    stood in for it. ``width`` is the crest's width as a fraction of the
    figure's width. ``zorder`` has to clear whatever the crest is drawn onto:
    the visuals' header strip is at 90, and at the default the crests were
    painted underneath it and vanished.
    """
    image = crest_image(team_id, allow_download=allow_download)
    if text_colour is None:
        # The monogram sits on the kit colour, so it follows the fill.
        text_colour = "#111418" if _relative_luminance(
            mcolors.to_rgb(colour)) > 0.42 else "#FFFFFF"
    width_frac = width
    height_frac = width * fig.get_figwidth() / fig.get_figheight()

    if image is None:
        # Monogram roundel: an ellipse in figure fractions, since the figure is
        # not square and a circle in those units would not be round.
        fig.add_artist(Ellipse((x, y), width_frac, height_frac, facecolor=colour,
                               edgecolor="none", zorder=zorder))
        fig.add_artist(Ellipse((x, y), width_frac * 0.78, height_frac * 0.78,
                               facecolor="none", edgecolor=text_colour, lw=1.1,
                               alpha=0.5, zorder=zorder + 1))
        fig.text(x, y, monogram, color=text_colour,
                 fontsize=width * fig.get_figwidth() * 8.3,
                 fontweight="bold", ha="center", va="center", zorder=zorder + 2)
        return False

    if needs_plate(image, background):
        pad_w, pad_h = width_frac * 0.14, height_frac * 0.14
        fig.add_artist(FancyBboxPatch(
            (x - width_frac / 2 - pad_w, y - height_frac / 2 - pad_h),
            width_frac + 2 * pad_w, height_frac + 2 * pad_h,
            boxstyle="round,pad=0,rounding_size=0.008",
            facecolor=plate_colour(image, background), edgecolor="none",
            alpha=0.93, zorder=zorder - 1,
        ))

    _draw(fig, image, x, y, width, zorder=zorder)
    return True


LOGO_PATH = ROOT / "assets" / "logo.jpg"
_LOGO_MEMO: list = []


def logo_image() -> np.ndarray | None:
    """The publisher's badge, or None on a clone that does not carry it."""
    if not _LOGO_MEMO:
        _LOGO_MEMO.append(_load(LOGO_PATH) if LOGO_PATH.exists() else None)
    return _LOGO_MEMO[0]


def place_logo(fig, x: float, y: float, *, width: float,
               background: str = "#000000") -> bool:
    """Draw the publisher's badge at a figure-fraction position.

    The badge is a JPEG on its own black ground, so on a light page it lands as
    a bare black square. It gets a rounded plate of the same black there and
    reads as a deliberate badge tile, the way the report's cover does.
    """
    image = logo_image()
    if image is None:
        return False
    width_frac = width
    height_frac = width * fig.get_figwidth() / fig.get_figheight()
    if _relative_luminance(mcolors.to_rgb(background)) > 0.5:
        pad_w, pad_h = width_frac * 0.09, height_frac * 0.09
        fig.add_artist(FancyBboxPatch(
            (x - width_frac / 2 - pad_w, y - height_frac / 2 - pad_h),
            width_frac + 2 * pad_w, height_frac + 2 * pad_h,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor="#0A0A0A", edgecolor="none", zorder=94,
        ))
    _draw(fig, image, x, y, width, zorder=95)
    return True


def prefetch(team_ids) -> dict[int, bool]:
    """Warm the cache for a fixture. Returns which ids resolved to a crest."""
    return {int(tid): crest_image(int(tid)) is not None for tid in team_ids}
