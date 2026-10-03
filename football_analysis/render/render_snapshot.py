"""Offline re-render of a saved fixture, without scraping or external writes."""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from football_analysis.pipeline.package_io import transactional_package


@transactional_package
def refresh_publication(
    events, players, xg, team_metrics, player_metrics, match_info, output_dir, *, source
):
    """Reuse saved boards, regenerate the analytical extension and publications.

    This path is only correct while the chart set has not changed. It copies
    the saved PNGs forward and rebuilds a named handful, so a release that adds
    a chart, drops one or renumbers the series must use the full package build
    instead: run without ``--publication-only``.

    Refreshing an out-of-date folder produced a package holding both series at
    once -- 07_xt_map beside 08_xt_map, 39_pitch_control beside
    44_pitch_control, sixty-eight files where the current set has fifty -- and
    the deleted charts survived while the new ones never appeared. The guard
    below makes that a refusal rather than a corrupt folder.
    """
    import shutil
    from football_analysis.visuals.insight_visuals import build_insight_visuals
    from football_analysis.visuals.visual_redesign_full import configure_match, build_catalog

    out = Path(output_dir)
    saved = {p.name for p in Path(source).glob("[0-9]*.png")}
    for p in Path(source).glob("*.png"):
        shutil.copy2(p, out / p.name)
    for name, frame in [
        ("players", players),
        ("xg", xg),
        ("team_advanced_metrics", team_metrics),
        ("player_sequence_metrics", player_metrics),
    ]:
        frame.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
    (out / "match_info.json").write_text(
        json.dumps(match_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    configure_match(match_info, out)
    # Rebuild the boards whose calculations or labels changed.
    from football_analysis.visuals import visual_redesign_full as visual

    visual.xt_map(events, match_info["home_id"], 7)
    visual.xt_map(events, match_info["away_id"], 8)
    visual.control_surface(events)
    visual.win_probability_curve(events)
    visual.shot_map(events, xg, match_info["home_id"], 2)
    visual.shot_map(events, xg, match_info["away_id"], 3)
    visual.turnovers(events, match_info["home_id"], 37)
    visual.turnovers(events, match_info["away_id"], 38)
    new, _ = build_insight_visuals(events, players, match_info, out)
    paths = sorted(set(out.glob("[0-9]*.png")) | set(new))
    # Two files claiming the same number means the copied series and the
    # rebuilt one are different series, and the folder now holds both. The
    # numbers are the reading order for the PDF and the catalogue, so a
    # package with a collision is not publishable. Refuse rather than write it.
    numbers = {}
    for path in paths:
        numbers.setdefault(path.name.split("_")[0], []).append(path.name)
    clashes = {key: sorted(value) for key, value in numbers.items() if len(value) > 1}
    if clashes:
        raise RuntimeError(
            "The saved chart set is not the one this build produces, so the refresh "
            "would publish both at once: "
            + "; ".join(f"{key} -> {', '.join(value)}" for key, value in sorted(clashes.items()))
            + ". Rebuild the whole package instead (drop --publication-only)."
        )
    missing = saved - {p.name for p in paths}
    if missing:
        raise RuntimeError(
            "Charts were copied forward that this build no longer produces: "
            + ", ".join(sorted(missing))
            + ". Rebuild the whole package instead (drop --publication-only)."
        )
    build_catalog([p.resolve() for p in paths])
    from football_analysis.reports.tactical_pdf_report import build_tactical_pdf
    from football_analysis.prose.match_article import build_match_article

    pdf = build_tactical_pdf(
        paths,
        out / "full_visual_redesign_real_data.pdf",
        events,
        xg,
        team_metrics,
        player_metrics,
        match_info,
    )
    article = build_match_article(
        events, xg, team_metrics, player_metrics, match_info, out, players
    )
    from football_analysis.visuals.poster_dashboard import build_match_posters

    for pattern in ("match_poster_*.png", "thread_*.png"):
        for old in out.glob(pattern):
            old.unlink()
    posters = build_match_posters(
        events,
        xg,
        team_metrics,
        player_metrics,
        players,
        out_dir=out,
        home_id=match_info["home_id"],
        away_id=match_info["away_id"],
        home_name=match_info["home_name"],
        away_name=match_info["away_name"],
        home_color=visual.HOME,
        away_color=visual.AWAY,
        score=match_info["score"],
        match_date=match_info.get("date", ""),
    )
    return {
        "output_dir": out,
        "pdf": pdf,
        "article": article,
        "visuals": paths,
        "posters": posters,
        "catalog": out / "visual_catalog.csv",
    }


# Team metrics added after a package was written. They are deterministic
# functions of the saved events, so a snapshot fills them in rather than
# publishing a board that reads "unavailable" for something the match has.
_LATE_TEAM_COLUMNS = ("xGoT", "line_breaking_passes", "line_breaking_completed")


def _backfill_team_metrics(events, info, team_frame):
    missing = [c for c in _LATE_TEAM_COLUMNS if c not in team_frame.columns]
    if not missing:
        return team_frame
    from football_analysis.metrics.match_metrics import advanced_metrics_frames

    fresh, _players = advanced_metrics_frames(events, info)
    by_side = fresh.set_index("side")
    team_frame = team_frame.copy()
    for column in missing:
        team_frame[column] = team_frame["side"].map(by_side[column])
    return team_frame


def load_snapshot(source):
    source = Path(source)
    info = json.loads((source / "match_info.json").read_text(encoding="utf-8-sig"))
    names = ["events", "players", "xg", "team_advanced_metrics", "player_sequence_metrics"]
    events, players, xg, team_metrics, player_metrics = (
        pd.read_csv(source / (name + ".csv")) for name in names
    )
    team_metrics = _backfill_team_metrics(events, info, team_metrics)
    return events, players, xg, team_metrics, player_metrics, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=["light", "dark"], default="dark")
    parser.add_argument("--both-themes", action="store_true")
    parser.add_argument(
        "--publication-only",
        action="store_true",
        help="Reuse saved boards in the selected theme; rebuild changed boards and publications",
    )
    args = parser.parse_args()
    os.environ["MATCH_ANALYSIS_THEME"] = args.theme
    os.environ["MATCH_ANALYSIS_LIGHT_COPY"] = "1" if args.both_themes else "0"
    from football_analysis.visuals.visual_redesign_full import generate_match_package

    if args.publication_only:
        result = refresh_publication(*load_snapshot(args.source), args.output, source=args.source)
    else:
        result = generate_match_package(*load_snapshot(args.source), args.output)
        if args.both_themes and not result.get("light_output_dir"):
            raise RuntimeError(
                "The requested light package did not finish; inspect the render log."
            )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
