"""Collect Opta's per-shot xG for every fixture we have published.

Two things read these shot maps. ``fit_xg_reference.py`` fits the engine's
correction layer to them, so a fixture FotMob does not cover is still priced
as close to Opta as the event feed allows; and ``reference_xg`` hands a
fixture's values straight to the render and to ``refresh_xg.py``, which read
the stored map before they ever reach for the network.

    python scripts/collect_reference_xg.py             # fetch what is missing
    python scripts/collect_reference_xg.py --dry-run   # match fixtures, fetch nothing

Each fixture's shot map is written to output/reference_xg/fotmob/<id>.json.gz
(output/ is not tracked), and output/reference_xg/fixtures.csv maps every
package to the FotMob fixture it was matched with. A run picks up where the
last one stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import reference_xg as RX  # noqa: E402

OUTPUT = ROOT / "output"
STORE = OUTPUT / "reference_xg"


def our_packages() -> list[dict]:
    rows = []
    for path in sorted(OUTPUT.rglob("match_info.json")):
        folder = path.parent
        if folder.name == "light" or any(p.startswith(".") for p in folder.parts):
            continue
        info = json.loads(path.read_text(encoding="utf-8-sig"))
        rows.append({"package": str(folder.relative_to(OUTPUT)).replace("\\", "/"),
                     "competition": info.get("competition"),
                     "date": str(info.get("date"))[:10],
                     "home": info.get("home_name"), "away": info.get("away_name"),
                     "score": str(info.get("score") or "")})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="match fixtures, fetch no shot maps")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between page loads")
    args = parser.parse_args()

    from curl_cffi import requests as cr

    session = cr.Session()
    packages = our_packages()
    by_competition: dict[str, list[dict]] = {}
    for competition in sorted({p["competition"] for p in packages}):
        if competition not in RX.LEAGUES:
            print(f"no FotMob league for {competition!r}")
            continue
        by_competition[competition] = RX.league_fixtures(session, competition)
        print(f"{competition}: {len(by_competition[competition])} finished on FotMob")
        time.sleep(args.delay)

    rows, unmatched = [], []
    for package in packages:
        fixture = RX.match_fixture(package, by_competition.get(package["competition"], []))
        if fixture is None:
            unmatched.append(package["package"])
            continue
        rows.append({**package, "fotmob_id": fixture["id"], "fotmob_url": fixture["url"],
                     "fotmob_home": fixture["home"], "fotmob_away": fixture["away"],
                     "_fixture": fixture})

    fetched = skipped = failed = 0
    for row in rows:
        if RX.cached(row["fotmob_id"]) is not None:
            skipped += 1
            continue
        if args.dry_run:
            continue
        payload = RX.fetch_shotmap(session, row["_fixture"])
        if payload is None:
            failed += 1
            print(f"  ! no shot map for {row['package']}")
        else:
            RX.save(payload, row["package"])
            fetched += 1
            print(f"  {row['package']}  ->  {len(payload['shots'])} shots")
        time.sleep(args.delay + random.uniform(0, 1.5))

    STORE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "_fixture"} for r in rows]).to_csv(
        STORE / "fixtures.csv", index=False, encoding="utf-8-sig")
    print(f"\nmatched {len(rows)}/{len(packages)} packages | fetched {fetched} "
          f"| already had {skipped} | failed {failed}"
          + (" (dry run)" if args.dry_run else ""))
    if unmatched:
        print("unmatched:", *unmatched, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
