"""The display face: a condensed bold for titles and big numbers.

Barlow Condensed (SIL Open Font License, see ``assets/fonts/OFL.txt``) ships
with the project, so a board looks the same on a laptop, in CI and on a server.
Body text stays in DejaVu Sans, which matplotlib carries itself. If the font
files are missing the display face falls back to DejaVu Sans Bold and every
board still draws, only wider.
"""

from __future__ import annotations

from matplotlib import font_manager

from football_analysis.paths import ASSETS_DIR

FONT_DIR = ASSETS_DIR / "fonts"
CONDENSED = "Barlow Condensed"
BODY = "DejaVu Sans"

_registered: bool | None = None


def register() -> bool:
    """Add the bundled files to matplotlib once; say whether the face is available."""
    global _registered
    if _registered is None:
        files = sorted(FONT_DIR.glob("BarlowCondensed-*.ttf"))
        for path in files:
            font_manager.fontManager.addfont(str(path))
        _registered = bool(files)
    return _registered


def display_family() -> str:
    """Family name to pass as ``fontfamily`` for titles and large figures."""
    return CONDENSED if register() else BODY


def display(size: float, weight: str = "bold") -> dict:
    """Keyword arguments for ``text``: condensed bold at ``size`` points.

    The condensed face sets about a quarter narrower than DejaVu at the same
    size, so callers pass the larger size they want it to read at.
    """
    return {"fontfamily": display_family(), "fontsize": size, "fontweight": weight}
