"""
match_store.py
═════════════════════════════════════════════════════════════════════════════
Persistent history of every analysed match.

Each run of the analyzer writes to its own output folder and then forgets the
match. That makes any question spanning more than one game impossible to
answer — a team's last six, a player's season shape, or a percentile measured
against anything wider than the twenty-two players on the pitch that day.

This module keeps a single SQLite file alongside the outputs and appends one
row per team per match and one row per player per match. SQLite is used rather
than a pile of CSVs because the useful questions are filters and aggregates
("this team, last 6, average field tilt"), and because re-analysing the same
fixture has to replace its previous rows rather than double-count them.

Typical use
───────────
    from match_store import save_match, team_match_log, team_totals

    save_match(info, team_frame, player_frame, url=MATCH_URL)
    recent = team_match_log("Arsenal", limit=6)
    summary = team_totals("Arsenal", limit=6)
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent / "output" / "match_history.db"

# Untouched provider payloads live beside the database. Keeping them means a
# new metric can be computed across every stored match without fetching a
# single page again — the source is asked once, ever.
RAW_DIRNAME = "raw_snapshots"

# Minimum team-match rows before a percentile is reported at all.
MIN_PERCENTILE_SAMPLE = 10

# Columns kept out of the metric blob because they identify the row instead of
# describing performance.
_IDENTITY_COLUMNS = {"team", "team_id", "side", "player", "match_id"}


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    _init_schema(connection)
    return connection


def _init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS matches (
            match_id     TEXT PRIMARY KEY,
            played_on    TEXT,
            competition  TEXT,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            home_goals   INTEGER,
            away_goals   INTEGER,
            score        TEXT,
            url          TEXT,
            raw_path     TEXT,
            stored_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS team_match_stats (
            match_id   TEXT NOT NULL,
            team       TEXT NOT NULL,
            opponent   TEXT NOT NULL,
            side       TEXT,
            goals_for  INTEGER,
            goals_against INTEGER,
            metrics    TEXT NOT NULL,
            PRIMARY KEY (match_id, team),
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS player_match_stats (
            match_id TEXT NOT NULL,
            player   TEXT NOT NULL,
            team     TEXT,
            metrics  TEXT NOT NULL,
            PRIMARY KEY (match_id, player),
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_match_stats(team);
        CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_match_stats(player);
        CREATE INDEX IF NOT EXISTS idx_matches_played_on ON matches(played_on);
        """
    )
    # CREATE TABLE IF NOT EXISTS leaves an older database on its original
    # columns, so any column added after the first release has to be applied
    # separately or every insert against an existing file fails.
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(matches)")}
    for column, definition in (("raw_path", "TEXT"),):
        if column not in existing:
            connection.execute(f"ALTER TABLE matches ADD COLUMN {column} {definition}")
    connection.commit()


def match_identity(info: dict, url: str | None = None) -> str:
    """Return a stable id for a fixture.

    The provider's own match id is used when the URL carries one, because it
    survives a team being renamed and it is what makes a re-run replace the
    earlier rows instead of adding a duplicate.

    The fallback deliberately does **not** include the date. A postponed
    fixture is dated differently by different sources — and by the same source
    before and after the move — so a date in the key turns one match into two
    rows: the events attach to one, the score to the other, and the team's
    history quietly splits. In league play each ordered pair of teams meets
    once per season, so competition, season and the two teams identify it
    uniquely and survive the reschedule. The date is kept as data on the row,
    where it can be checked, not as part of the key.
    """
    if url:
        found = re.search(r"/matches/(\d+)", str(url))
        if found:
            return f"ws-{found.group(1)}"

    home = str(info.get("home_name") or "home").strip().lower().replace(" ", "-")
    away = str(info.get("away_name") or "away").strip().lower().replace(" ", "-")
    competition = str(info.get("competition") or "").strip().lower().replace(" ", "-")
    season = str(info.get("season") or "").strip().lower().replace(" ", "-")
    parts = [part for part in (competition, season, f"{home}-vs-{away}") if part]
    return "-".join(parts)


def _metric_blob(row: dict) -> str:
    payload = {
        key: value
        for key, value in row.items()
        if key not in _IDENTITY_COLUMNS and not isinstance(value, (dict, list))
    }
    return json.dumps(payload, default=str)


def _goals(info: dict) -> tuple[int | None, int | None]:
    numbers = re.findall(r"\d+", str(info.get("score") or ""))
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    home = info.get("home_score")
    away = info.get("away_score")
    try:
        return int(home), int(away)
    except (TypeError, ValueError):
        return None, None


def save_match(
    info: dict,
    team_metrics: pd.DataFrame,
    player_metrics: pd.DataFrame | None = None,
    *,
    url: str | None = None,
    db_path: Path | str | None = None,
) -> str:
    """Append one match to the history, replacing it if it is already stored.

    Returns the match id it was stored under.
    """
    match_id = match_identity(info, url)
    home = str(info.get("home_name") or "Home")
    away = str(info.get("away_name") or "Away")
    home_goals, away_goals = _goals(info)

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO matches (match_id, played_on, competition, home_team, away_team,
                                 home_goals, away_goals, score, url, stored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                played_on=excluded.played_on, competition=excluded.competition,
                home_team=excluded.home_team, away_team=excluded.away_team,
                home_goals=excluded.home_goals, away_goals=excluded.away_goals,
                score=excluded.score, url=excluded.url, stored_at=excluded.stored_at
            """,
            (
                match_id,
                str(info.get("date") or info.get("played_on") or "")[:10] or None,
                str(info.get("competition") or "") or None,
                home,
                away,
                home_goals,
                away_goals,
                str(info.get("score") or "") or None,
                str(url or "") or None,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )

        # A re-run must not leave the previous rows behind alongside the new ones.
        connection.execute("DELETE FROM team_match_stats WHERE match_id = ?", (match_id,))
        connection.execute("DELETE FROM player_match_stats WHERE match_id = ?", (match_id,))

        if team_metrics is not None and not team_metrics.empty:
            for row in team_metrics.to_dict("records"):
                side = str(row.get("side") or "")
                team = str(row.get("team") or (home if side == "home" else away))
                opponent = away if team == home else home
                for_goals = home_goals if team == home else away_goals
                against_goals = away_goals if team == home else home_goals
                connection.execute(
                    """
                    INSERT INTO team_match_stats
                        (match_id, team, opponent, side, goals_for, goals_against, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (match_id, team, opponent, side or None, for_goals, against_goals, _metric_blob(row)),
                )

        if player_metrics is not None and not player_metrics.empty:
            for row in player_metrics.to_dict("records"):
                player = str(row.get("player") or "").strip()
                if not player:
                    continue
                connection.execute(
                    """
                    INSERT INTO player_match_stats (match_id, player, team, metrics)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(match_id, player) DO UPDATE SET metrics=excluded.metrics
                    """,
                    (match_id, player, str(row.get("team") or "") or None, _metric_blob(row)),
                )
        connection.commit()
    return match_id



def _raw_dir(db_path: Path | str | None = None) -> Path:
    return Path(db_path or DEFAULT_DB).parent / RAW_DIRNAME


def save_snapshot(match_id: str, payload: Any, db_path: Path | str | None = None) -> Path:
    """Store a provider payload verbatim, gzipped, before anything parses it.

    Written as-is on purpose: the moment it is normalised it inherits today's
    parser, and the point of keeping it is to survive a parser change.
    """
    directory = _raw_dir(db_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{match_id}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), default=str)

    with _connect(db_path) as connection:
        connection.execute(
            "UPDATE matches SET raw_path = ? WHERE match_id = ?", (path.name, match_id)
        )
        connection.commit()
    return path


def load_snapshot(match_id: str, db_path: Path | str | None = None) -> Any | None:
    """Return a stored payload, or None if this match was never snapshotted."""
    path = _raw_dir(db_path) / f"{match_id}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def stored_snapshots(db_path: Path | str | None = None) -> list[str]:
    """Return the match ids that can be replayed offline."""
    directory = _raw_dir(db_path)
    if not directory.exists():
        return []
    return sorted(path.name[: -len(".json.gz")] for path in directory.glob("*.json.gz"))


def _expand(rows: Iterable[sqlite3.Row]) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {key: row[key] for key in row.keys() if key != "metrics"}
        record.update(json.loads(row["metrics"]))
        records.append(record)
    return pd.DataFrame(records)


def list_matches(team: str | None = None, db_path: Path | str | None = None) -> pd.DataFrame:
    """Return every stored match, newest first, optionally filtered to one team."""
    with _connect(db_path) as connection:
        if team:
            cursor = connection.execute(
                """
                SELECT * FROM matches
                WHERE home_team = ? COLLATE NOCASE OR away_team = ? COLLATE NOCASE
                ORDER BY COALESCE(played_on, stored_at) DESC
                """,
                (team, team),
            )
        else:
            cursor = connection.execute(
                "SELECT * FROM matches ORDER BY COALESCE(played_on, stored_at) DESC"
            )
        return pd.DataFrame([dict(row) for row in cursor.fetchall()])


def team_match_log(team: str, limit: int | None = None, db_path: Path | str | None = None) -> pd.DataFrame:
    """Return one row per stored match for a team, newest first."""
    query = """
        SELECT s.*, m.played_on, m.competition, m.score, m.url
        FROM team_match_stats s
        JOIN matches m ON m.match_id = s.match_id
        WHERE s.team = ? COLLATE NOCASE
        ORDER BY COALESCE(m.played_on, m.stored_at) DESC
    """
    with _connect(db_path) as connection:
        rows = connection.execute(query, (team,)).fetchall()
    frame = _expand(rows)
    if limit is not None and not frame.empty:
        frame = frame.head(int(limit))
    return frame.reset_index(drop=True)


def team_totals(team: str, limit: int | None = None, db_path: Path | str | None = None) -> pd.DataFrame:
    """Return a team's per-match average and total for every numeric metric.

    Rates and percentages are meaningless when summed, so both views are
    returned side by side and the caller picks the one that fits the metric
    rather than the function guessing.
    """
    log = team_match_log(team, limit=limit, db_path=db_path)
    if log.empty:
        return pd.DataFrame(columns=["metric", "matches", "total", "per_match"])

    numeric = log.select_dtypes(include="number")
    numeric = numeric.drop(columns=[c for c in ("team_id",) if c in numeric.columns], errors="ignore")
    return pd.DataFrame(
        {
            "metric": numeric.columns,
            "matches": len(log),
            "total": [round(float(numeric[c].sum()), 3) for c in numeric.columns],
            "per_match": [round(float(numeric[c].mean()), 3) for c in numeric.columns],
        }
    ).reset_index(drop=True)


def player_match_log(player: str, limit: int | None = None, db_path: Path | str | None = None) -> pd.DataFrame:
    """Return one row per stored match for a player, newest first."""
    query = """
        SELECT p.*, m.played_on, m.competition, m.home_team, m.away_team, m.score
        FROM player_match_stats p
        JOIN matches m ON m.match_id = p.match_id
        WHERE p.player = ? COLLATE NOCASE
        ORDER BY COALESCE(m.played_on, m.stored_at) DESC
    """
    with _connect(db_path) as connection:
        rows = connection.execute(query, (player,)).fetchall()
    frame = _expand(rows)
    if limit is not None and not frame.empty:
        frame = frame.head(int(limit))
    return frame.reset_index(drop=True)


def metric_percentile(
    metric: str, value: float, *, team: str | None = None, db_path: Path | str | None = None
) -> float | None:
    """Return where a value sits against every stored match for that metric.

    This is what the radar pages have been missing: a percentile measured
    against a real history rather than against the other players on the pitch
    that afternoon. Returns None until there is enough stored history for the
    answer to mean anything.
    """
    with _connect(db_path) as connection:
        if team:
            rows = connection.execute(
                "SELECT metrics FROM team_match_stats WHERE team = ? COLLATE NOCASE", (team,)
            ).fetchall()
        else:
            rows = connection.execute("SELECT metrics FROM team_match_stats").fetchall()

    values = []
    for row in rows:
        payload = json.loads(row["metrics"])
        if metric in payload:
            try:
                values.append(float(payload[metric]))
            except (TypeError, ValueError):
                continue
    # Counted in team-match rows, not fixtures: every match contributes two
    # rows, so this is five fixtures' worth. Below that a percentile is noise
    # dressed as a number, and silence is the honest answer.
    if len(values) < MIN_PERCENTILE_SAMPLE:
        return None
    series = pd.Series(values)
    return round(float((series <= float(value)).mean() * 100), 1)


__all__ = [
    "DEFAULT_DB",
    "MIN_PERCENTILE_SAMPLE",
    "list_matches",
    "match_identity",
    "metric_percentile",
    "player_match_log",
    "load_snapshot",
    "save_match",
    "save_snapshot",
    "stored_snapshots",
    "team_match_log",
    "team_totals",
]
