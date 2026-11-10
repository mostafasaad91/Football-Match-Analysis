"""Cross-check a built package: article, PDF and boards must tell one story.

Every number the article quotes is recomputed here with the same function the
board beside it is drawn from, and every claim that can contradict the data
("the better chances", "never enough to force an equaliser") is checked
against it. A package passes only when the article, the PDF and the boards
agree.

    python scripts/verify_package.py <package dir> [...]        # one line per package
    python scripts/verify_package.py --all output                # every package under output/
    python scripts/verify_package.py --all output --json report.json

Exit status is 1 when any package has a problem.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


@dataclass
class Report:
    folder: str
    ok: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        (self.ok if cond else self.problems).append(f"{name}" + (f": {detail}" if detail else ""))


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def _num(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _section(body: str, heading: str) -> str:
    """The text under one heading, or '' when the article has no such heading."""
    if heading not in body:
        return ""
    return body.split(heading, 1)[1].split("\n## ", 1)[0]


def _sentence(body: str, *needles: str) -> str:
    """The first sentence holding every needle, or ''."""
    for s in re.split(r"(?<=[.!?])\s+", body):
        if all(n in s for n in needles):
            return s
    return ""


def verify(folder: Path) -> Report:
    from football_analysis.metrics.expected_threat import rescore
    from football_analysis.metrics.match_extras import match_extras
    from football_analysis.metrics.match_metrics import (
        box_entry_mask,
        build_possessions,
        high_regain_events,
        team_advanced_metrics,
    )
    from football_analysis.reports.match_report import calculate_ppda

    rep = Report(str(folder))
    md_path = folder / "match_article.md"
    if not md_path.exists():
        rep.check("article exists", False, "match_article.md missing")
        return rep
    info = json.loads((folder / "match_info.json").read_text(encoding="utf-8-sig"))
    H, A = info["home_name"], info["away_name"]
    HID, AID = info["home_id"], info["away_id"]
    name_of = {HID: H, AID: A}
    events = rescore(pd.read_csv(folder / "events.csv", low_memory=False))
    xg = pd.read_csv(folder / "xg.csv")
    md = md_path.read_text(encoding="utf-8")
    body, _, guide = md.partition("## Complete visual guide")

    # ---- Score and xG -------------------------------------------------------
    goals_txt = re.findall(r"\d+", str(info.get("score", "")))
    if len(goals_txt) < 2:
        rep.check("score readable", False, str(info.get("score")))
        return rep
    hs, as_ = int(goals_txt[0]), int(goals_txt[1])
    rep.check("score", f"{max(hs, as_)}–{min(hs, as_)}" in body or f"{hs}–{as_}" in body, info["score"])
    xg_of = {}
    for _, r in xg.iterrows():
        xg_of[str(r["team"])] = float(r["xG"])
        rep.check(f"xG {r['team']}", f"{float(r['xG']):.2f}" in body, f"{float(r['xG']):.2f}")

    # ---- Shots as the shot maps draw them (events) --------------------------
    shots = events[_truthy(events["is_shot"])]
    if "is_own_goal" in shots.columns:
        shots = shots[~_truthy(shots["is_own_goal"])]
    if "is_penalty_shootout" in shots.columns:
        shots = shots[~_truthy(shots["is_penalty_shootout"])]
    for tid, name in name_of.items():
        mine = shots[shots["team_id"].eq(tid)]
        raw = mine.get("shot_whoscored_type", mine["type"]).astype(str)
        n, on = len(mine), int(raw.isin(["Goal", "SavedShot"]).sum())
        took = re.search(rf"{re.escape(name)} took (\d+) shots: (\d+) on target", guide)
        if took:
            rep.check(f"shot map caption {name}", (int(took[1]), int(took[2])) == (n, on),
                      f"caption {took[1]}/{took[2]}, events {n}/{on}")
        row = xg[xg["team"].astype(str).str.lower().eq(name.lower())]
        if len(row) and "on_target" in row:
            rep.check(f"xg.csv shot counts {name} match the events",
                      (int(row["shots"].iloc[0]), int(row["on_target"].iloc[0])) == (n, on),
                      f"xg.csv {int(row['shots'].iloc[0])}/{int(row['on_target'].iloc[0])}, events {n}/{on}")

    # ---- Possession, PPDA ---------------------------------------------------
    tm = team_advanced_metrics(events, info)
    for side, tid, oid in (("home", HID, AID), ("away", AID, HID)):
        name = name_of[tid]
        pos = round(float(tm[side]["possession_share"]))
        if "% of the ball" in body or "% of possession" in body:
            rep.check(f"possession {name}", f"{pos}%" in body, f"{pos}%")
        ppda = calculate_ppda(events, tid, oid)["ppda"]
        if ppda:
            rep.check(f"PPDA {name}", f"{ppda:.1f}" in body, f"{ppda:.1f}")

    # ---- Box entries: board count, article, halves ---------------------------
    for side, tid in (("home", HID), ("away", AID)):
        name = name_of[tid]
        own = events[events["team_id"].eq(tid)]
        n = int(box_entry_mask(own).sum())
        quoted = re.search(rf"(?:They got into the box|They reached the penalty area) (\d+) times",
                           _section(body, f"## {name} with the ball"))
        if quoted:
            rep.check(f"box entries {name} (article = board)", int(quoted[1]) == n, f"article {quoted[1]}, board {n}")
        halves = re.findall(rf"{re.escape(name)} played \d+ passes in the (?:first|second) half.*?(\d+) entr(?:y|ies) into the box", guide)
        if len(halves) == 2:
            rep.check(f"box entries {name}: halves add up", sum(int(h) for h in halves) == n,
                      f"{halves[0]} + {halves[1]} vs {n}")

    # ---- High regains: count and what followed (board title) -----------------
    annotated, possessions = build_possessions(events)
    by_id = possessions.set_index("possession_id") if not possessions.empty else None
    for side, tid in (("home", HID), ("away", AID)):
        name = name_of[tid]
        frame = high_regain_events(events, tid).dropna(subset=["x", "y"])
        led = 0
        if by_id is not None and len(frame):
            led = int((annotated.loc[frame.index, "possession_id"].map(by_id["shots"]).fillna(0) > 0).sum())
        section = _section(body, f"## {name} without the ball")
        m = re.search(r"(?:won the ball back high|high up the pitch) (\d+) times(?:, and (\d+) of those regains led to a shot)?", section)
        old = re.search(r"(?:won the ball back|took the ball off .+?) (\d+) times in [^.,]*"
                        r"(?:, and their attacks after winning it produced (\d+) shots?)?", section)
        if m:
            rep.check(f"high regains {name} (article = board)", int(m[1]) == len(frame), f"article {m[1]}, board {len(frame)}")
            rep.check(f"high regains to shot {name}", int(m[2] or 0) == led, f"article {m[2] or 0}, board {led}")
        elif old:
            rep.check(f"high regains {name} (article = board)", int(old[1]) == len(frame), f"article {old[1]}, board {len(frame)}")
            if old[2] is not None:
                rep.check(f"high regains to shot {name}", int(old[2]) == led,
                          f"article pairs {old[2]} transition shots with the regains, board {led}")

    # ---- Board titles, when the build recorded them --------------------------
    record = folder / "board_text.json"
    if record.exists():
        boards = json.loads(record.read_text(encoding="utf-8"))
        for file, texts in boards.items():
            joined = " ".join(texts)
            for tid, name in name_of.items():
                m = re.search(rf"{re.escape(name)} got into the box (\d+) times? · (\d+) led to a shot", joined)
                if m and "box_entries" in file:
                    n = int(box_entry_mask(events[events["team_id"].eq(tid)]).sum())
                    rep.check(f"board {file} box entries", int(m[1]) == n, f"board {m[1]}, data {n}")
                    q = re.search(r"(?:They got into the box|They reached the penalty area) (\d+) times",
                                  _section(body, f"## {name} with the ball"))
                    if q:
                        rep.check(f"board {file} = article", q[1] == m[1], f"board {m[1]}, article {q[1]}")
                m = re.search(rf"{re.escape(name)} won the ball back high (\d+) times? · (\d+) led to a shot", joined)
                if m and "high_regains" in file:
                    q = re.search(r"(?:won the ball back high|high up the pitch) (\d+) times(?:, and (\d+) of those)?",
                                  _section(body, f"## {name} without the ball"))
                    if q:
                        rep.check(f"board {file} = article", (q[1], q[2] or "0") == (m[1], m[2]),
                                  f"board {m[1]}/{m[2]}, article {q[1]}/{q[2] or 0}")

    # ---- Claims that can contradict the data ---------------------------------
    hx, ax = xg_of.get(H, 0.0), xg_of.get(A, 0.0)
    winner = H if hs > as_ else A if as_ > hs else None
    if winner:
        loser = A if winner == H else H
        wx, lx = (hx, ax) if winner == H else (ax, hx)
        if lx > wx:
            for phrase in ("turned that territory into the better chances", "better chances and the composure"):
                rep.check(f"no '{phrase}' when {loser} had more xG", phrase not in body, f"{lx:.2f} vs {wx:.2f}")
    early = _sentence(body, "Because it came so early")
    if early:
        m = re.search(r"the match became (.+?) with the ball and (.+?) without", early)
        if m:
            sides = {H: ("home", HID), A: ("away", AID)}
            if m[1] in sides and m[2] in sides:
                rep.check("early-goal frame matches possession",
                          tm[sides[m[1]][0]]["possession_share"] > tm[sides[m[2]][0]]["possession_share"],
                          f"says {m[1]} had the ball")
    goals = shots[_truthy(shots.get("is_goal", pd.Series(False, index=shots.index)))] if "is_goal" in shots else shots.iloc[0:0]
    for tid, name in name_of.items():
        s = _sentence(body, f"{name} produced", "never enough to force an equaliser")
        if s and "Chasing the game" in s:
            # did they score while behind?
            all_goals = events[_truthy(events["is_goal"])] if "is_goal" in events else events.iloc[0:0]
            score = {HID: 0, AID: 0}
            came_back = False
            for _, g in all_goals.sort_values(["period_code", "minute", "second"] if "second" in all_goals else ["minute"]).iterrows():
                scorer = g["team_id"]
                if _truthy(pd.Series([g.get("is_own_goal", False)])).iloc[0]:
                    scorer = AID if scorer == HID else HID
                if scorer in score:
                    if scorer == tid and score[tid] < score[AID if tid == HID else HID]:
                        came_back = True
                    score[scorer] += 1
            rep.check(f"'never enough to force an equaliser' for {name}", not came_back, "they did score while behind")

    # ---- Extras --------------------------------------------------------------
    try:
        ex = match_extras(events, info)
        xp = ex["expected_points"]
        s = _sentence(body, "expected points to")
        if s:
            rep.check("expected points", f"{xp['home_xpts']:.1f} expected points to {xp['away_xpts']:.1f}" in s
                      or f"{xp['away_xpts']:.1f} expected points to {xp['home_xpts']:.1f}" in s, s[:120])
        for key, name in (("home", H), ("away", A)):
            m = re.search(rf"{re.escape(name)} took (\d+) corners?", body)
            if m:
                rep.check(f"corners {name}", int(m[1]) == ex[key]["set_pieces"]["corners"],
                          f"article {m[1]}, data {ex[key]['set_pieces']['corners']}")
    except Exception as error:  # noqa: BLE001
        rep.check("match extras computed", False, repr(error))

    # ---- Writing hygiene ------------------------------------------------------
    for pattern, label in ((r"\b0\.00 per possession", "0.00 per possession"), (r"\bnan\b", "nan"),
                           (r"\bNone\b", "None"), (r"\b1 (shots|entries|goals|chances|corners|saves|regains)\b", "plural after 1"),
                           (r"\b(\w{3,}) \1\b", "repeated word")):
        hits = re.findall(pattern, body)
        rep.check(f"no '{label}'", not hits, str(hits[:3]))

    # ---- The PDF carries the article -----------------------------------------
    pdf = folder / "full_visual_redesign_real_data.pdf"
    try:
        import fitz

        doc = fitz.open(pdf)
        text = re.sub(r"\s+", " ", " ".join(page.get_text() for page in doc))
        doc.close()
        nums = set(re.findall(r"\b\d+\.\d{2}\b", body))
        missing = sorted(n for n in nums if n not in text)
        rep.check("PDF carries every 2-decimal figure of the article", not missing, f"missing {missing[:8]}")
        title = md.splitlines()[0].lstrip("# ").strip()
        rep.check("PDF is this article", title[:30] in text, title)
    except Exception as error:  # noqa: BLE001
        rep.check("PDF readable", False, repr(error))
    return rep


def packages(root: Path) -> list[Path]:
    found = [p.parent for p in root.rglob("match_info.json")]
    return sorted(p for p in found if not any(part.startswith(".") or part == "_wk" for part in p.parts))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--all", metavar="ROOT")
    parser.add_argument("--json", metavar="FILE")
    parser.add_argument("--light", action="store_true", help="include light/ copies")
    args = parser.parse_args(argv)
    folders = [Path(p) for p in args.paths]
    if args.all:
        folders += [p for p in packages(Path(args.all)) if args.light or p.name != "light"]
    results, bad = [], 0
    for folder in folders:
        try:
            rep = verify(folder)
        except Exception:  # noqa: BLE001
            rep = Report(str(folder), problems=["verifier crashed: " + traceback.format_exc(limit=2).replace("\n", " | ")])
        bad += bool(rep.problems)
        results.append({"folder": rep.folder, "ok": len(rep.ok), "problems": rep.problems})
        status = "OK  " if not rep.problems else "FAIL"
        print(f"{status} {len(rep.ok):3d} ok {len(rep.problems):2d} bad  {folder}", flush=True)
        for p in rep.problems:
            print(f"       ✗ {p}", flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(folders) - bad} of {len(folders)} packages clean")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
