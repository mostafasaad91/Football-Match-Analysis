"""Where the project keeps what it reads and writes.

Modules live two levels down in the package, so none of them can find the
output folder, the assets or the data from their own location any more; they
ask here instead.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
# Fitted coefficients the xG layers read at render time.
MODELS_DIR = DATA_DIR / "models"
