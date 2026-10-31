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
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Four times a normal fixture, so a slow provider or a big match is never cut
# off, and a hung scrape costs half an hour instead of the rest of the round.
# Counted in time the machine was awake: see _run_child.
FIXTURE_TIMEOUT_S = 30 * 60
# How often a running fixture is looked at. A gap between two looks much longer
# than this is the machine asleep, and is not charged to the fixture.
POLL_S = 5.0
ASLEEP_AFTER_S = 60.0
MATCH_URL = re.compile(
    r"https?://(?:www\.)?whoscored\.com/matches/\d+/(?:live|show)/\S*", re.IGNORECASE
)


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
                    + (
                        f"Saved rounds in rounds/: {', '.join(available)}"
                        if available
                        else "No saved rounds in rounds/ yet."
                    )
                )
        found.extend(MATCH_URL.findall(path.read_text(encoding="utf-8")))
    ordered, seen = [], set()
    for url in found:
        # /show/ and /live/ address the same fixture; the pipeline wants /live/.
        url = url.replace("/show/", "/live/").rstrip("/,")
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _round_folder(url: str, environment: dict) -> Path:
    from football_analysis.pipeline.match_fixture import shelf

    where = ROOT / "output"
    for part in shelf(url, environment.get("MATCH_ANALYSIS_ROUND", "")):
        where = where / part
    return where


def _words(text: str) -> list[str]:
    """``Atlético_Madrid`` and ``atletico-madrid`` both become ``atletico madrid``."""
    plain = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in plain if not unicodedata.combining(c)).lower()
    return [word for word in re.split(r"[^a-z0-9]+", plain) if word]


def _names_team(team: str, slug: list[str]) -> bool:
    """Whether a folder's team can be one of the clubs the URL spells out.

    The folder carries the feed's short name and the URL the club's full one,
    so ``Man_Utd`` has to be found in ``manchester-united`` and ``PSG`` in
    ``paris-saint-germain``.
    """
    wanted = _words(team)
    if not wanted:
        return False
    if "".join(wanted) in "".join(slug):
        return True

    def shortens(short: str, word: str) -> bool:
        rest = iter(word)
        return short[0] == word[0] and all(letter in rest for letter in short)

    at = 0
    for short in wanted:
        while at < len(slug) and not shortens(short, slug[at]):
            at += 1
        if at == len(slug):
            break
        at += 1
    else:
        return True
    initials = "".join(word[0] for word in slug)
    return len(wanted) == 1 and len(wanted[0]) > 1 and wanted[0] in initials


def _fixture_package(url: str, environment: dict, since: float = 0.0) -> Path | None:
    """This fixture's published package, or None when it cannot be told apart.

    The folder is named from the teams and the score, which this process never
    parsed, but the URL ends in the same two clubs: ``...-2026-2027-genoa-
    juventus`` against ``Genoa_vs_Juventus_2-0``. Two rounds run side by side
    shelve into the same folder, so "the newest package" can be the other
    process's fixture; the teams cannot. A package whose manifest is older
    than ``since`` was left by an earlier run, not this one.
    """
    where = _round_folder(url, environment)
    if not where.is_dir():
        return None
    # The clubs are what follows the season; "parma-calcio-1913" keeps its year.
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    slug = _words(re.split(r"-\d{4}(?:-\d{4})?-", tail, maxsplit=1)[-1])
    found = []
    for package in where.iterdir():
        manifest = package / "package_manifest.json"
        if package.name.startswith(".") or not manifest.is_file():
            continue
        # A file's time is stamped from a coarser clock than time.time() reads.
        if manifest.stat().st_mtime < since - 2:
            continue
        home, _, away = re.sub(r"_\d+-\d+$", "", package.name).partition("_vs_")
        if away and _names_team(home, slug) and _names_team(away, slug):
            found.append(package)
    # More than one is the same tie shelved twice under different scores.
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _has_boards(folder: Path) -> bool:
    return folder.is_dir() and any(folder.glob("[0-9]*.png"))


def _light_missing(url: str, environment: dict) -> Path | None:
    """The package this fixture just wrote, when its light copy did not arrive.

    Found by its teams when they can be read off the URL, and otherwise by the
    one thing that is certain: it is the newest package under the round it was
    shelved in.
    """
    package = _fixture_package(url, environment)
    if package is None:
        where = _round_folder(url, environment)
        if not where.is_dir():
            return None
        packages = [
            p
            for p in where.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / "match_info.json").exists()
        ]
        if not packages:
            return None
        package = max(packages, key=lambda p: p.stat().st_mtime)
    light = package / "light"
    if light.is_dir() and list(light.glob("*.png")):
        return None
    return package


def _stop_tree(process: subprocess.Popen) -> None:
    """Stop the fixture and everything it started.

    Killing only the fixture left its light render and its Chrome running on,
    orphaned, drawing into a staging folder nobody would publish while the
    next fixture competed with them for the machine.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def _run_child(command: list[str], environment: dict, limit: float) -> dict:
    """Run one child to the end, or stop it after ``limit`` seconds awake.

    This used to be subprocess.run(capture_output=True, timeout=...), and two
    Serie A fixtures that had published both themes were reported "no result
    after 30 min, stopped". The laptop had gone into Modern Standby for
    sixteen minutes mid-render. Windows does not count standby against a
    wait's timeout but the monotonic clock does, so the half hour never ran
    out while the fixture was alive; when it exited, run() made its one last
    zero-second wait on a process that had closed its pipes but not yet
    finished exiting, and called a finished fixture a timeout.

    So the fixture is looked at every POLL_S, whether it has finished is asked
    before whether its time is up, and a gap between looks of more than
    ASLEEP_AFTER_S is the machine asleep, charged to nobody. Its output goes
    to files rather than pipes: Chrome inherits the fixture's handles, and a
    pipe is not at its end until the last process holding it lets go.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        started = last = time.monotonic()
        awake = 0.0
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=out,
            stderr=err,
            # Its own group, so a stop reaches whatever it started.
            start_new_session=os.name != "nt",
        )
        timed_out = False
        while True:
            try:
                process.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            step = now - last
            last = now
            if step <= ASLEEP_AFTER_S:
                awake += step
            if process.poll() is not None:
                break
            if awake >= limit:
                _stop_tree(process)
                timed_out = True
                break
        wall = time.monotonic() - started

        def text(handle) -> str:
            handle.seek(0)
            return handle.read().decode("utf-8", errors="replace")

        return {
            "returncode": None if timed_out else process.returncode,
            "stdout": text(out),
            "stderr": text(err),
            "wall": wall,
            "asleep": max(0.0, wall - awake) if wall - awake > ASLEEP_AFTER_S else 0.0,
        }


def _took(child: dict) -> str:
    took = f"{child['wall'] / 60:.1f} min"
    if child["asleep"]:
        took += f" ({child['asleep'] / 60:.1f} min of it asleep)"
    return took


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
    # A fixture renders in about seven minutes. One Serie A fixture sat for
    # three hours on six seconds of CPU, waiting on the provider mid-scrape,
    # and with no limit here the whole round waited behind it. Past the limit
    # the child and everything it started are stopped and reported, and the
    # round moves on.
    finished = _run_child(
        [sys.executable, "football_match_analysis.py"], environment, FIXTURE_TIMEOUT_S
    )
    took = _took(finished)
    if finished["returncode"] is None:
        # The package is published in one rename after both themes are drawn,
        # so a manifest written since this fixture started, with boards in
        # both themes beside it, is a finished fixture however the clock read.
        package = _fixture_package(url, environment, since=started)
        if (
            package is not None
            and _has_boards(package)
            and (dark_only or _has_boards(package / "light"))
        ):
            return True, f"{took} :: past the {FIXTURE_TIMEOUT_S // 60} min limit, package complete"
        return False, f"{took} :: no result after {FIXTURE_TIMEOUT_S // 60} min, stopped"
    if finished["returncode"] == 0:
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
            retry = _run_child(
                [
                    sys.executable,
                    "-m",
                    "football_analysis.render.render_light",
                    str(missing),
                    "--child",
                ],
                {
                    **environment,
                    "MATCH_ANALYSIS_THEME": "light",
                    "MATCH_ANALYSIS_LIGHT_COPY": "0",
                },
                FIXTURE_TIMEOUT_S,
            )
            if retry["returncode"] != 0:
                tail = (retry["stderr"] or retry["stdout"] or "").strip().splitlines()[-2:]
                return True, took + " :: light copy still missing :: " + " | ".join(
                    line.strip() for line in tail
                )
            took += " (+light retried)"
        return True, took
    # The last few lines carry the reason; the whole log is rarely the point.
    tail = (finished["stderr"] or finished["stdout"] or "").strip().splitlines()[-3:]
    return False, f"{took} :: " + " | ".join(line.strip() for line in tail)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--round", required=True, help='Matchweek number or name: 1, "Matchweek 1", "الجولة 1"'
    )
    parser.add_argument("--url", action="append", help="One fixture URL; repeatable")
    parser.add_argument("--urls", action="append", help="A file holding fixture URLs; repeatable")
    parser.add_argument(
        "--dark-only", action="store_true", help="Skip the light copy (it is built by default)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would run, and where it would be shelved"
    )
    parser.add_argument(
        "--no-refit", action="store_true", help="Skip refitting the xG layer after the round"
    )
    args = parser.parse_args()

    from football_analysis.pipeline.match_fixture import normalise_round, shelf

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
    if len(failures) < len(urls) and not args.no_refit:
        refit_xg()
    return 1 if failures else 0


def refit_xg() -> None:
    """Refit the xG layers on every Opta shot map stored so far.

    Each fixture rendered keeps its shot map, so the archive the layers learn
    from grows by a round at a time, and refitting here is what makes them
    improve over the season rather than stay where they were fitted: the
    pre-shot layer on top of the engine, and the post-shot model goalkeepers
    are judged against. Each script writes nothing unless its new fit holds up.
    """
    jobs = (
        (
            "xG layer",
            "scripts/fit_xg_reference.py",
            ("shots paired", "Opta-fitted", "what ships", "The fit"),
        ),
        ("post-shot xG", "scripts/fit_psxg.py", ("on-target shots", "fitted (new)", "The fit")),
    )
    for name, script, shown in jobs:
        print(f"\nRefitting the {name} on the stored Opta shot maps ...", flush=True)
        try:
            done = subprocess.run(
                [sys.executable, script],
                cwd=ROOT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=FIXTURE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            print("  refit took too long, stopped; the previous fit stays")
            continue
        for line in (done.stdout or done.stderr or "").strip().splitlines():
            if any(token in line for token in shown + ("written",)):
                print("  " + line.strip())


if __name__ == "__main__":
    raise SystemExit(main())
