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



def _light_missing(url: str, environment: dict) -> Path | None:
    """The package this fixture just wrote, when its light copy did not arrive.

    The renderer names the folder from the teams and the score, none of which
    this process parsed, so the fixture is found by the one thing that is
    certain: it is the newest package under the round it was shelved in.
    """
    from match_fixture import shelf

    where = ROOT / "output"
    for part in shelf(url, environment.get("MATCH_ANALYSIS_ROUND", "")):
        where = where / part
    if not where.is_dir():
        return None
    packages = [p for p in where.iterdir()
                if p.is_dir() and not p.name.startswith(".")
                and (p / "match_info.json").exists()]
    if not packages:
        return None
    newest = max(packages, key=lambda p: p.stat().st_mtime)
    light = newest / "light"
    if light.is_dir() and list(light.glob("*.png")):
        return None
    return newest


def run_one(url: str, round_name: str, dark_only: bool) -> tuple[bool, str]:
    environment = dict(os.environ)
    environment["MATCH_ANALYSIS_URL"] = url
    environment["MATCH_ANALYSIS_ROUND"] = round_name
    # The package builds the light copy unless told otherwise - the default in
    # visual_redesign_full is "1". Writing "0" here turned it off for every
    # fixture in a round, silently, because the flag looked like it was opting
    # in to something rather than out of it. Only an explicit --dark-only says
    # anything now; otherwise the pipeline's own default stands.
    if dark_only:
        environment["MATCH_ANALYSIS_LIGHT_COPY"] = "0"
    # The pipeline prints Arabic, and text=True decodes with the console's
    # code page, which on Windows is cp1252. The reader thread then dies with
    # a UnicodeDecodeError and the child's output is lost - so a fixture that
    # failed reported no reason at all. Read it as UTF-8 and never raise on a
    # byte that is not: a mangled character in a log is a smaller problem than
    # a missing log.
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.time()
    finished = subprocess.run([sys.executable, "football_match_analysis.py"],
                              cwd=ROOT, env=environment, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    took = f"{(time.time() - started) / 60:.1f} min"
    if finished.returncode == 0:
        # The light copy is a second full render, launched by the fixture's own
        # process while it still holds every figure it drew. CPython does not
        # give those pages back, so on a machine without several spare gigabytes
        # matplotlib dies allocating the child's first canvas and the package
        # ships with a dark half and no light one -- which looks finished from
        # the outside. This process holds nothing, so a retry from here gets the
        # clean allocation the first attempt could not. Nothing happens when the
        # light copy is already there, which is the usual case.
        missing = None if dark_only else _light_missing(url, environment)
        if missing is not None:
            retry = subprocess.run(
                [sys.executable, "render_light.py", str(missing), "--child"],
                cwd=ROOT, env={**environment, "MATCH_ANALYSIS_THEME": "light",
                               "MATCH_ANALYSIS_LIGHT_COPY": "0"},
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if retry.returncode != 0:
                tail = (retry.stderr or retry.stdout or "").strip().splitlines()[-2:]
                return True, took + " :: light copy still missing :: " + " | ".join(
                    line.strip() for line in tail)
            took += " (+light retried)"
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
    parser.add_argument("--dark-only", action="store_true",
                        help="Skip the light copy (it is built by default)")
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
        ok, detail = run_one(url, round_name, args.dark_only)
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
