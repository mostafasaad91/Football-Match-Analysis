"""The bundled display face: present, registered, and safe to lose."""

from matplotlib import font_manager

from football_analysis.visuals import typography


def test_the_condensed_face_ships_with_its_licence():
    assert (typography.FONT_DIR / "OFL.txt").exists()
    assert sorted(p.name for p in typography.FONT_DIR.glob("BarlowCondensed-*.ttf"))


def test_the_display_face_resolves_to_the_bundled_file():
    family = typography.display_family()
    assert family == typography.CONDENSED
    found = font_manager.findfont(
        font_manager.FontProperties(family=family, weight="bold"), fallback_to_default=False
    )
    assert "BarlowCondensed" in found


def test_display_kwargs_are_a_bold_text_style():
    style = typography.display(20)
    assert style["fontsize"] == 20 and style["fontweight"] == "bold"
    assert style["fontfamily"] == typography.display_family()


def test_a_missing_font_directory_falls_back_to_the_body_face(monkeypatch, tmp_path):
    monkeypatch.setattr(typography, "FONT_DIR", tmp_path)
    monkeypatch.setattr(typography, "_registered", None)
    assert typography.display_family() == typography.BODY
    monkeypatch.setattr(typography, "_registered", None)
