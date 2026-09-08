"""Run every fixture of one matchweek, shelving them all under that round.

WhoScored does not publish a matchweek anywhere the pipeline can read it: not
in the match URL, not in the feed. The stage page groups fixtures by date, so
"round three" is only ever something the operator knows. That is why the round
is an input here rather than something inferred.

    python run_round.py --round 1 --urls urls.txt
    python run_round.py --round 1 --url https://... --url https://...

Each fixture runs in its own process, because the renderers keep module-level
state that one match configures and never puts back. A failure is reported and
the round carries on; the summary at the end says which ones need another go.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATCH_URL = re.compile(r"https?://(?:www\.)?whoscored\.com/matches/\d+/(?:live|show)/\S*",
                       re.IGNORECASE)


def read_urls(args) -> list[str]:
    """Every fixture URL the caller gave, in order, without duplicates.

    A file is read for anything that looks like a match URL rather than parsed
    line by line, so a list pasted out of the browser works whether it arrives
    one per line, comma separated, or wrapped in quotes.
    """
    found: list[str] = list(args.url or [])
    for name in args.urls or []:
        path = Path(name)
        if not path.is_file():
            # A bare filename is almost always one of the saved round lists, so
            # look there before giving up. A traceback out of pathlib says the
            # file is missing without saying where it was looked for.
            beside = ROOT / "rounds" / path.name
            if beside.is_file():
                path = beside
            else:
                available = sorted(p.name for p in (ROOT / "rounds").glob("*.txt"))
                raise SystemExit(
                    f"No URL file at {name!r}.\n"
                    + (f"Saved rounds in rounds/: {', '.join(available)}"
                       if available else "No saved rounds in rounds/ yet."))
        found.extend(MATCH_URL.findall(path.read_text(encoding="utf-8")))
    ordered, seen = [], set()
    for url in found:
        # /show/ and /live/ address the same fixture; the pipeline wants /live/.
        url = url.replace("/show/", "/live/").rstrip("/,")
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def run_one(url: str, round_name: str, both_themes: bool) -> tuple[bool, str]:
    environment = dict(os.environ)
    environment["MATCH_ANALYSIS_URL"] = url
    environment["MATCH_ANALYSIS_ROUND"] = round_name
    environment["MATCH_ANALYSIS_LIGHT_COPY"] = "1" if both_themes else "0"
    started = time.time()
    finished = subprocess.run([sys.executable, "football_match_analysis.py"],
                              cwd=ROOT, env=environment,
                              capture_output=True, text=True)
    took = f"{(time.time() - started) / 60:.1f} min"
    if finished.returncode == 0:
        return True, took
    # The last few lines carry the reason; the whole log is rarely the point.
    tail = (finished.stderr or finished.stdout or "").strip().splitlines()[-3:]
    return False, f"{took} :: " + " | ".join(line.strip() for line in tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", required=True,
                        help='Matchweek number or name: 1, "Matchweek 1", "الجولة 1"')
    parser.add_argument("--url", action="append", help="One fixture URL; repeatable")
    parser.add_argument("--urls", action="append",
                        help="A file holding fixture URLs; repeatable")
    parser.add_argument("--both-themes", action="store_true",
                        help="Render the light copy beside the dark one")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would run, and where it would be shelved")
    args = parser.parse_args()

    from match_fixture import normalise_round, shelf

    round_name = normalise_round(args.round) or normalise_round(f"Matchweek {args.round}")
    if not round_name:
        print(f"Cannot read a round from {args.round!r}", file=sys.stderr)
        return 2

    urls = read_urls(args)
    if not urls:
        print("No fixture URLs given. Use --url or --urls.", file=sys.stderr)
        return 2

    where = shelf(urls[0], round_name)
    print(f"{round_name}  ·  {len(urls)} fixtures  ->  output/{'/'.join(where)}/")
    for index, url in enumerate(urls, start=1):
        print(f"  {index:2d}. {url.rsplit('/', 1)[-1]}")
    if args.dry_run:
        return 0

    print()
    failures = []
    for index, url in enumerate(urls, start=1):
        label = url.rsplit("/", 1)[-1]
        print(f"[{index}/{len(urls)}] {label} ... ", end="", flush=True)
        ok, detail = run_one(url, round_name, args.both_themes)
        print("ok " + detail if ok else "FAILED " + detail)
        if not ok:
            failures.append((url, detail))

    print(f"\n{len(urls) - len(failures)}/{len(urls)} finished.")
    if failures:
        print("Re-run these:")
        for url, detail in failures:
            print(f"  {url}\n      {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
