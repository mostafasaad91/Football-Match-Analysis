"""run_round tells a finished fixture from a hung one.

Two Serie A fixtures that had published both themes were reported "no result
after 30 min, stopped": the laptop slept mid-render, and the timeout only
fired as the fixture exited. These run tiny child scripts, never a package.
"""

import importlib.util
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

RUN_ROUND = Path(__file__).resolve().parent.parent / "run_round.py"
GENOA = "https://www.whoscored.com/matches/1981050/live/italy-serie-a-2026-2027-genoa-juventus"


@pytest.fixture
def rr(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_round_under_test", RUN_ROUND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ is not "__main__", so nothing runs
    monkeypatch.setattr(module, "POLL_S", 0.1)
    return module


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(code)]


def test_a_child_that_finished_is_ok_even_when_its_time_is_up(rr, monkeypatch):
    # The limit has already run out when the child exits: what it did is
    # asked before how long it took, so it is a result, not a timeout.
    monkeypatch.setattr(rr, "POLL_S", 30.0)
    child = rr._run_child(_python("print('published')"), dict(os.environ), limit=0)
    assert child["returncode"] == 0
    assert "published" in child["stdout"]


def test_a_hung_child_is_stopped_with_everything_it_started(rr, tmp_path):
    marker = tmp_path / "orphan_wrote.txt"
    grandchild = (
        f"import time, pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('x')"
    )
    child = _python(
        f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {grandchild!r}])
        time.sleep(120)
        """
    )
    started = time.monotonic()
    result = rr._run_child(child, dict(os.environ), limit=1.0)
    assert result["returncode"] is None
    assert time.monotonic() - started < 30
    # The light render the fixture had started must not draw on, orphaned.
    time.sleep(4)
    assert not marker.exists()


def test_a_grandchild_holding_the_output_does_not_hold_the_round(rr):
    # Chrome inherits the fixture's handles and can outlive it. With the
    # output on a pipe, the round waited for Chrome rather than the fixture.
    child = _python(
        """
        import subprocess, sys
        held = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        print(held.pid, flush=True)
        """
    )
    started = time.monotonic()
    result = rr._run_child(child, dict(os.environ), limit=60)
    try:
        assert result["returncode"] == 0
        assert time.monotonic() - started < 15
    finally:
        pid = int(result["stdout"].split()[0])
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def test_time_the_machine_was_asleep_is_not_charged_to_the_fixture(rr, monkeypatch):
    real = time.monotonic
    calls = {"n": 0}

    def clock():
        # Twenty minutes of standby land between two looks at the child.
        calls["n"] += 1
        return real() + (1200 if calls["n"] > 2 else 0)

    monkeypatch.setattr(rr.time, "monotonic", clock)
    result = rr._run_child(_python("import time; time.sleep(1.5)"), dict(os.environ), limit=60)
    assert result["returncode"] == 0
    assert result["asleep"] > 1000
    assert "asleep" in rr._took(result)


def _package(root: Path, name: str, light: bool = True) -> Path:
    folder = root / "output" / "Italy_Serie_A" / "2026-2027" / "Matchweek_09" / name
    folder.mkdir(parents=True)
    (folder / "01_xg_flow.png").write_bytes(b"png")
    if light:
        (folder / "light").mkdir()
        (folder / "light" / "01_xg_flow.png").write_bytes(b"png")
    (folder / "package_manifest.json").write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture
def timed_out(rr, monkeypatch, tmp_path):
    """run_one with a child that is stopped at its limit, after publishing what is queued."""
    monkeypatch.setattr(rr, "ROOT", tmp_path)
    rr.publishes = []

    def child(*args, **kwargs):
        for publish in rr.publishes:
            publish()
        return {"returncode": None, "stdout": "", "stderr": "", "wall": 1890.0, "asleep": 0.0}

    monkeypatch.setattr(rr, "_run_child", child)
    return rr


def test_a_timed_out_fixture_whose_package_is_complete_is_ok(timed_out, tmp_path):
    timed_out.publishes.append(lambda: _package(tmp_path, "Genoa_vs_Juventus_2-0"))
    ok, detail = timed_out.run_one(GENOA, "Matchweek 09", dark_only=False)
    assert ok, detail
    assert detail.startswith("31.5 min") and "package complete" in detail


def test_a_timed_out_fixture_with_nothing_on_disk_is_reported(timed_out):
    ok, detail = timed_out.run_one(GENOA, "Matchweek 09", dark_only=False)
    assert not ok
    assert "no result after 30 min, stopped" in detail


def test_the_other_rounds_package_does_not_count(timed_out, tmp_path):
    # Two rounds run side by side shelve into one folder.
    timed_out.publishes.append(lambda: _package(tmp_path, "Torino_vs_Como_1-3"))
    ok, _ = timed_out.run_one(GENOA, "Matchweek 09", dark_only=False)
    assert not ok


def test_a_package_left_by_an_earlier_run_does_not_count(timed_out, tmp_path):
    old = _package(tmp_path, "Genoa_vs_Juventus_2-0")
    hour_ago = time.time() - 3600
    os.utime(old / "package_manifest.json", (hour_ago, hour_ago))
    ok, _ = timed_out.run_one(GENOA, "Matchweek 09", dark_only=False)
    assert not ok


def test_a_package_without_its_light_copy_is_not_complete(timed_out, tmp_path):
    timed_out.publishes.append(lambda: _package(tmp_path, "Genoa_vs_Juventus_2-0", light=False))
    assert not timed_out.run_one(GENOA, "Matchweek 09", dark_only=False)[0]
    manifest = next(tmp_path.rglob("package_manifest.json"))
    timed_out.publishes[:] = [lambda: manifest.write_text("{}", encoding="utf-8")]
    assert timed_out.run_one(GENOA, "Matchweek 09", dark_only=True)[0]


@pytest.mark.parametrize(
    "folder, slug",
    [
        ("Genoa_vs_Juventus_2-0", "italy-serie-a-2026-2027-genoa-juventus"),
        ("Parma_Calcio_1913_vs_Udinese_0-0", "italy-serie-a-2026-2027-parma-calcio-1913-udinese"),
        (
            "Atletico_vs_Man_Utd_2-0",
            "europe-champions-league-2026-2027-atletico-madrid-manchester-united",
        ),
        (
            "Man_City_vs_PSG_2-4",
            "europe-champions-league-2026-2027-manchester-city-paris-saint-germain",
        ),
        ("Bayern_vs_BodoeGlimt_5-0", "europe-champions-league-2026-2027-bayern-munich-bodoe-glimt"),
    ],
)
def test_the_package_is_found_by_the_clubs_in_the_url(rr, monkeypatch, tmp_path, folder, slug):
    monkeypatch.setattr(rr, "ROOT", tmp_path)
    url = f"https://www.whoscored.com/matches/1/live/{slug}"
    environment = {"MATCH_ANALYSIS_ROUND": "Matchweek 09"}
    shelf = rr._round_folder(url, environment)
    for name in (folder, "Torino_vs_Como_1-3"):
        (shelf / name).mkdir(parents=True)
        (shelf / name / "package_manifest.json").write_text("{}", encoding="utf-8")
    assert rr._fixture_package(url, environment).name == folder
