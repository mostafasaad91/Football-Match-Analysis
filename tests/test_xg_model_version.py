"""A package is labelled with the xG model its shots were priced by."""

import json

import pandas as pd

from football_analysis.pipeline.package_io import transactional_package, xg_model_version


def _events(*sources):
    rows = [{"is_shot": True, "xg_source": s} for s in sources]
    rows.append({"is_shot": False, "xg_source": "not a shot"})
    return pd.DataFrame(rows)


def test_the_version_is_what_the_shots_record():
    assert xg_model_version(_events("model_a", "model_a")) == "model_a"


def test_a_mixed_package_says_so():
    assert xg_model_version(_events("model_a", "provider_shot_xg")) == (
        "model_a | provider_shot_xg"
    )


def test_no_source_means_no_version():
    assert xg_model_version(pd.DataFrame({"is_shot": [True]})) is None
    assert xg_model_version(_events("", None)) is None


def test_flags_read_back_from_csv_still_count():
    events = pd.DataFrame({"is_shot": ["True", "False"], "xg_source": ["model_a", "model_b"]})
    assert xg_model_version(events) == "model_a"


def test_the_render_sees_the_stamp_and_the_manifest_keeps_it(tmp_path):
    seen = {}

    @transactional_package
    def render(events, match_info, output_dir):
        seen["version"] = match_info.get("xg_model_version")
        (output_dir / "01_board.png").write_bytes(b"png")
        return {}

    render(_events("model_a"), {"competition": "Test"}, tmp_path / "match")
    manifest = json.loads((tmp_path / "match" / "package_manifest.json").read_text("utf-8"))
    assert seen["version"] == "model_a"
    assert manifest["fixture"]["xg_model_version"] == "model_a"
    assert manifest["models"]["xG"] == "model_a"
