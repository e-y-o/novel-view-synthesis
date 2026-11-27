"""Tests for nvs.core.exif (EXIF -> focal length). Uses tiny synthetic images only; no real photos."""

import math
from pathlib import Path

import pytest
from PIL import Image, ImageOps
from PIL.TiffImagePlugin import IFDRational

from nvs.core.exif import extract_image_meta

EXIF_IFD = 0x8769
ORIENTATION = 0x0112
FOCAL_LENGTH = 0x920A
F35 = 0xA405
FP_XRES = 0xA20E
FP_UNIT = 0xA210
PX_X, PX_Y = 0xA002, 0xA003
ZOOM = 0xA404


def make_jpeg(tmp_path: Path, size=(64, 48), ifd0=None, exif_ifd=None, name="img.jpg", fmt="JPEG") -> Path:
    exif = Image.Exif()
    for k, v in (ifd0 or {}).items():
        exif[k] = v
    if exif_ifd:
        exif[EXIF_IFD] = dict(exif_ifd)
    path = tmp_path / name
    Image.new("RGB", size, (128, 64, 32)).save(path, fmt, exif=exif.tobytes())
    return path


def test_35mm_rule(tmp_path):
    p = make_jpeg(tmp_path, (64, 48), exif_ifd={F35: 26})
    m = extract_image_meta(p)
    assert m.source == "35mm"
    assert (m.width, m.height) == (64, 48)
    assert m.f_px == pytest.approx(26 * math.hypot(64, 48) / 43.267)
    assert m.raw["FocalLengthIn35mmFilm"] == 26
    assert m.warnings == []


@pytest.mark.parametrize("unit,res", [(3, 100.0), (2, 254.0), (None, 254.0)])  # cm, inch, default=inch
def test_focal_plane_rule(tmp_path, unit, res):
    tags = {FOCAL_LENGTH: IFDRational(5, 1), FP_XRES: IFDRational(res)}
    if unit is not None:
        tags[FP_UNIT] = unit
    m = extract_image_meta(make_jpeg(tmp_path, (40, 30), exif_ifd=tags))
    # 10 px/mm -> sensor 4 mm wide for 40 px -> f_px = 5 mm * 40 px / 4 mm = 50
    assert m.source == "focal_plane"
    assert m.f_px == pytest.approx(50.0)
    assert m.raw["sensor_w_mm"] == pytest.approx(4.0)


def test_focal_plane_uses_exif_pixel_width_when_resized(tmp_path):
    # Tags describe an 80x60 frame; the file was downscaled to 40x30 -> f_px halves.
    tags = {FOCAL_LENGTH: IFDRational(5, 1), FP_XRES: IFDRational(100), FP_UNIT: 3, PX_X: 80, PX_Y: 60}
    m = extract_image_meta(make_jpeg(tmp_path, (40, 30), exif_ifd=tags))
    assert m.source == "focal_plane"
    assert m.f_px == pytest.approx(25.0)
    assert any(w.startswith("resized") for w in m.warnings)


def test_35mm_preferred_over_focal_plane(tmp_path):
    tags = {F35: 26, FOCAL_LENGTH: IFDRational(5, 1), FP_XRES: IFDRational(100), FP_UNIT: 3}
    m = extract_image_meta(make_jpeg(tmp_path, (40, 30), exif_ifd=tags))
    assert m.source == "35mm"
    assert "f_px_focal_plane" in m.raw  # still computed as a cross-check


def test_orientation_6_swaps_size(tmp_path):
    p = make_jpeg(tmp_path, (64, 48), ifd0={ORIENTATION: 6}, exif_ifd={F35: 26})
    m = extract_image_meta(p)
    assert m.orientation == 6
    assert (m.width, m.height) == (48, 64)
    with Image.open(p) as im:
        assert ImageOps.exif_transpose(im).size == (m.width, m.height)
    # Diagonal is rotation-invariant, so f_px matches the unrotated case.
    assert m.f_px == pytest.approx(26 * 80 / 43.267)


def test_orientation_1_keeps_size(tmp_path):
    m = extract_image_meta(make_jpeg(tmp_path, (64, 48), ifd0={ORIENTATION: 1}))
    assert m.orientation == 1
    assert (m.width, m.height) == (64, 48)


def test_missing_exif(tmp_path):
    p = tmp_path / "bare.jpg"
    Image.new("RGB", (64, 48)).save(p)
    m = extract_image_meta(p)
    assert (m.f_px, m.source, m.orientation) == (None, "none", None)
    assert (m.width, m.height) == (64, 48)
    assert m.raw == {}


def test_png_without_exif(tmp_path):
    p = tmp_path / "bare.png"
    Image.new("RGB", (32, 24)).save(p)
    assert extract_image_meta(p).source == "none"


def test_digital_zoom_warns(tmp_path):
    m = extract_image_meta(make_jpeg(tmp_path, exif_ifd={F35: 26, ZOOM: IFDRational(2, 1)}))
    assert m.source == "35mm"
    assert any("DigitalZoomRatio" in w for w in m.warnings)


@pytest.mark.parametrize("zoom", [IFDRational(1, 1), IFDRational(0, 1)])  # 0 = "not used" per EXIF
def test_no_digital_zoom_no_warning(tmp_path, zoom):
    m = extract_image_meta(make_jpeg(tmp_path, exif_ifd={F35: 26, ZOOM: zoom}))
    assert not any("DigitalZoomRatio" in w for w in m.warnings)


def test_crop_warns(tmp_path):
    # EXIF says 64x64, file is 64x48: aspect changed -> cropped.
    m = extract_image_meta(make_jpeg(tmp_path, (64, 48), exif_ifd={F35: 26, PX_X: 64, PX_Y: 64}))
    assert any(w.startswith("cropped") for w in m.warnings)


def test_rotated_exif_dimensions_not_flagged_as_crop(tmp_path):
    # Rotation baked into pixels and downscaled; tags still say landscape 80x60.
    m = extract_image_meta(make_jpeg(tmp_path, (30, 40), exif_ifd={F35: 26, PX_X: 80, PX_Y: 60}))
    assert not any(w.startswith("cropped") for w in m.warnings)
    assert any("rotated" in w for w in m.warnings)


def test_degenerate_values_do_not_crash(tmp_path):
    tags = {F35: 0, FOCAL_LENGTH: IFDRational(0, 0), FP_XRES: IFDRational(100), FP_UNIT: 7}
    m = extract_image_meta(make_jpeg(tmp_path, exif_ifd=tags))
    assert (m.f_px, m.source) == (None, "none")


def test_unknown_unit_skips_focal_plane(tmp_path):
    tags = {FOCAL_LENGTH: IFDRational(5, 1), FP_XRES: IFDRational(100), FP_UNIT: 7}
    m = extract_image_meta(make_jpeg(tmp_path, exif_ifd=tags))
    assert m.source == "none"
    assert any("FocalPlaneResolutionUnit" in w for w in m.warnings)


def test_heic(tmp_path):
    pytest.importorskip("pillow_heif")
    try:
        p = make_jpeg(tmp_path, (64, 48), exif_ifd={F35: 26}, name="img.heic", fmt="HEIF")
    except (KeyError, OSError, ValueError) as e:
        pytest.skip(f"no HEIF encoder: {e}")
    m = extract_image_meta(p)
    assert m.source == "35mm"
    assert (m.width, m.height) == (64, 48)
