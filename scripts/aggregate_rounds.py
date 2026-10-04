"""Add up a block of rounds into team and player tables.

    python scripts/aggregate_rounds.py                              # Premier League, rounds 1-5
    python scripts/aggregate_rounds.py --rounds 2-4 --min-minutes 180
    python scripts/aggregate_rounds.py --league Italy_Serie_A --rounds 1-5

Reads the packages already on disk (output/<league>/<season>/Matchweek_NN/<match>/)
and writes, under output/aggregates/<league>_<season>_R<from>-<to>/:

    teams_totals.csv      points, goals, and the sum of every count metric
    teams_per_match.csv   the average of every metric (the right view for rates)
    players_totals.csv    counts summed, ratios recomputed from the sums
    players_per90.csv     per-90 rates, for players with at least --min-minutes
    matches.csv           the matches that were added up
    aggregate.xlsx        the same tables as sheets

The round comes from the folder name, so a package filed under "Matchweek_05" is round
5 whatever label the history database holds for it. A fixture that appears in two round
folders is counted once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_analysis.metrics.aggregate import (  # noqa: E402
    aggregate_players,
    PLAYER_COUNTS,
    aggregate_teams,
    partial_columns,
    per90,
)
from football_analysis.metrics.extra_stats import player_extra, rating_by_player, team_extra  # noqa: E402
from football_analysis.paths import OUTPUT_DIR  # noqa: E402

CACHE_DIR = OUTPUT_DIR / "aggregates" / "_cache"


def parse_rounds(text: str) -> range:
    """``"1-5"`` -> rounds 1..5; ``"3"`` -> round 3."""
    match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", text)
    if not match:
        raise argparse.ArgumentTypeError(f"rounds must look like 1-5 or 3, not {text!r}")
    low = int(match.group(1))
    high = int(match.group(2) or low)
    if high < low:
        raise argparse.ArgumentTypeError("the last round is before the first")
    return range(low, high + 1)


def find_packages(league_dir: Path, rounds: range) -> list[tuple[int, Path]]:
    """Every package folder under the league's round folders, with its round number."""
    found, seen = [], set()
    for round_dir in sorted(league_dir.glob("Matchweek_*")):
        number = int(re.search(r"(\d+)$", round_dir.name).group(1))
        if number not in rounds:
            continue
        # Folders starting with a dot are the renderer's staging areas, copies of a
        # package that is still being written or was interrupted; they are never a match.
        for package in sorted(
            p
            for p in round_dir.iterdir()
            if not p.name.startswith(".") and (p / "match_info.json").exists()
        ):
            info = json.loads((package / "match_info.json").read_text(encoding="utf-8-sig"))
            key = str(info.get("match_id") or info.get("url") or package.name)
            if key in seen:
                continue
            seen.add(key)
            found.append((number, package))
    return found


def _score(info: dict) -> tuple[int, int]:
    goals = re.findall(r"\d+", str(info.get("score", "")))
    return (int(goals[0]), int(goals[1])) if len(goals) >= 2 else (0, 0)


def team_rows(package: Path, round_number: int, info: dict) -> pd.DataFrame:
    """One row per side: the xG table and the advanced metrics, with the result."""
    from football_analysis.render.render_snapshot import _backfill_team_metrics

    xg = pd.read_csv(package / "xg.csv")
    events = pd.read_csv(package / "events.csv", low_memory=False)
    # Packages rendered before xGOT and line-breaking passes were added lack those columns;
    # they are rebuilt from the saved events, as the offline renderer does.
    advanced = _backfill_team_metrics(
        events, info, pd.read_csv(package / "team_advanced_metrics.csv")
    )
    home_goals, away_goals = _score(info)
    out = []
    for side, name, goals_for, goals_against in (
        ("home", info["home_name"], home_goals, away_goals),
        ("away", info["away_name"], away_goals, home_goals),
    ):
        row = {"team": name, "match_id": package.name, "round": round_number}
        row.update({"goals_for": goals_for, "goals_against": goals_against})
        xg_row = xg[xg["team"].astype(str).eq(name)]
        if len(xg_row):
            row.update(
                {
                    k: v
                    for k, v in xg_row.iloc[0].items()
                    if k not in {"team", "goals"} and pd.api.types.is_number(v)
                }
            )
        adv_row = advanced[advanced["team"].astype(str).eq(name)]
        if len(adv_row):
            row.update(
                {
                    k: v
                    for k, v in adv_row.iloc[0].items()
                    if k not in {"team", "side", "team_id"}
                    and k not in row
                    and pd.api.types.is_number(v)
                }
            )
        out.append(row)
    frame = pd.DataFrame(out)
    frame = frame.merge(team_extra(events, info), on="team", how="left")

    # What each side conceded: the other row's attacking figures, so a table can say
    # who allowed the fewest chances and how well each goalkeeper did against the shots faced.
    against = {
        "xG": "xG_against",
        "xGoT": "xGoT_against",
        "shots": "shots_against",
        "on_target": "on_target_against",
        "big_chances": "big_chances_against",
        "box_entries": "box_entries_against",
        "key_passes": "key_passes_against",
    }
    for index in frame.index:
        other = frame.drop(index).iloc[0]
        for source, target in against.items():
            if source in frame:
                frame.loc[index, target] = other[source]
    return frame


def player_rows(package: Path, info: dict) -> pd.DataFrame:
    """The enriched per-player frame the match's cards use.

    Building it takes about ten seconds a match, so it is kept under
    output/aggregates/_cache and rebuilt only when the package's events change. The cache
    lives outside the package folder so the package stays exactly what the renderer wrote.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"v2__{package.parent.parent.parent.name}__{package.name}.csv"
    events_file = package / "events.csv"
    if cache.exists() and cache.stat().st_mtime >= events_file.stat().st_mtime:
        return pd.read_csv(cache)
    from football_analysis.render.render_snapshot import load_snapshot
    from football_analysis.visuals.advanced_profiles import _people_for

    events, players, _xg, _teams, _pm, snapshot_info = load_snapshot(package)
    people = _people_for(events, players, snapshot_info).copy()
    names = {info["home_id"]: info["home_name"], info["away_id"]: info["away_name"]}
    people["team"] = people["team_id"].map(names)
    people["match_id"] = package.name
    # Counts that are not in the card frame (assists, key passes, cards ...), and the
    # provider's rating and starting place, matched on player and team.
    people = people.merge(player_extra(events), on=["player", "team_id"], how="left")
    people = people.merge(rating_by_player(players), on=["player", "team_id"], how="left")
    extra = [c for c in player_extra(events).columns if c not in {"player", "team_id"}]
    people[extra + ["started"]] = people[extra + ["started"]].fillna(0)
    people.to_csv(cache, index=False, encoding="utf-8-sig")
    return people


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--league", default="England_Premier_League", help="folder under output/")
    parser.add_argument("--season", default="2026-2027")
    parser.add_argument("--rounds", type=parse_rounds, default=parse_rounds("1-5"))
    parser.add_argument(
        "--min-minutes", type=float, default=90.0, help="floor for players_per90.csv"
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "aggregates")
    args = parser.parse_args(argv)

    league_dir = OUTPUT_DIR / args.league / args.season
    packages = find_packages(league_dir, args.rounds)
    if not packages:
        print(
            f"No packages under {league_dir} for rounds {args.rounds.start}-{args.rounds.stop - 1}"
        )
        return 1

    teams, people, listing = [], [], []
    for number, package in packages:
        info = json.loads((package / "match_info.json").read_text(encoding="utf-8-sig"))
        try:
            teams.append(team_rows(package, number, info))
            people.append(player_rows(package, info))
        except Exception as error:  # one broken package must not lose the other forty-eight
            print(f"skipped {package.name}: {type(error).__name__}: {error}")
            continue
        listing.append(
            {
                "round": number,
                "match": package.name,
                "date": info.get("date"),
                "score": info.get("score"),
            }
        )
        print(f"R{number}  {package.name}")

    all_teams = pd.concat(teams, ignore_index=True)
    all_people = pd.concat(people, ignore_index=True)
    # Player ratios are blank by nature where nothing was attempted, and goals prevented
    # exists only for goalkeepers; the counts are what must be complete.
    player_counts = all_people[
        [c for c in PLAYER_COUNTS if c in all_people and c != "goals_prevented"]
    ]
    for label, frame in (("team", all_teams), ("player", player_counts)):
        gaps = partial_columns(frame)
        if gaps:
            print(
                f"WARNING: {label} columns missing for some matches, so their totals cover fewer: {gaps}"
            )
    team_totals, team_means = aggregate_teams(all_teams)
    player_totals = aggregate_players(all_people)
    player_rates = per90(player_totals, args.min_minutes)
    matches = pd.DataFrame(listing)

    first, last = args.rounds.start, args.rounds.stop - 1
    folder = args.out / f"{args.league}_{args.season}_R{first}-{last}"
    folder.mkdir(parents=True, exist_ok=True)
    tables = {
        "teams_totals": team_totals,
        "teams_per_match": team_means,
        "players_totals": player_totals,
        "players_per90": player_rates,
        "matches": matches,
    }
    for name, table in tables.items():
        table.round(3).to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(folder / "aggregate.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            table.round(3).to_excel(writer, sheet_name=name[:31], index=False)
    print(
        f"\n{len(listing)} matches, {len(team_totals)} teams, {len(player_totals)} players"
        f" ({len(player_rates)} with {args.min_minutes:.0f}+ minutes) -> {folder}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
