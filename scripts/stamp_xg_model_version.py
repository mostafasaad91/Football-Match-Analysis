"""Record which xG model priced each package already on disk.

The history charts compare a match only with packages priced by the same xG
model, and they read that from ``fixture.xg_model_version`` in each package's
manifest. Nothing wrote the field until now, so no package was ever comparable
with another and the charts were always skipped. New renders stamp it; this
stamps the packages rendered before, reading the version off each package's
own shots (``xg_source``) rather than assuming the current model priced them.

    python scripts/stamp_xg_model_version.py             # stamp every package
    python scripts/stamp_xg_model_version.py --dry-run   # count, write nothing

Only the manifest is touched. ``match_info.json`` is hashed in the manifest's
inputs, so it is left as it is.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from football_analysis.pipeline.package_io import xg_model_version  # noqa: E402

OUTPUT = ROOT / "output"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    args = parser.parse_args()

    stamped, skipped = collections.Counter(), 0
    for manifest_path in sorted(OUTPUT.rglob("package_manifest.json")):
        if any(part.startswith(".") for part in manifest_path.relative_to(OUTPUT).parts):
            continue
        # The light copy keeps its own manifest but prices from the same events.
        package = manifest_path.parent
        events_path = package / "events.csv"
        if not events_path.exists() and package.name == "light":
            events_path = package.parent / "events.csv"
        if not events_path.exists():
            skipped += 1
            continue
        events = pd.read_csv(
            events_path, usecols=lambda c: c in {"is_shot", "xg_source"}, low_memory=False
        )
        version = xg_model_version(events)
        if not version:
            skipped += 1
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        competition = (manifest.get("fixture") or {}).get("competition")
        stamped[(competition, version)] += 1
        if args.dry_run:
            continue
        manifest.setdefault("fixture", {})["xg_model_version"] = version
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    for (competition, version), count in stamped.most_common():
        print(f"{count:4d}  {competition}  ·  {version}")
    print(
        f"{sum(stamped.values())} manifest(s) {'would be ' if args.dry_run else ''}stamped, "
        f"{skipped} skipped (no shots or no events)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
