"""Preview asset packaging (T05): sizes, contract, filters, float16 precision, privacy, atomic publish."""

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from synth import render, scene_two_layers

from nvs.core.camera import make_camera
from nvs.core.exif import ImageMeta
from nvs.core.types import DepthResult, Intrinsics
from nvs.preview import assets
from nvs.preview.assets import (
    PreviewError,
    exif_summary,
    package_preview,
    preview_size,
    resample_color,
    resample_depth,
    resample_sky,
)
from nvs.preview.layout import ASSET_NAMES, validate_dir

FG, BG = 2.0, 6.0


def make_inputs(W=2400, H=1600, sky_rows=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(3, H, W, generator=g)
    v = torch.linspace(0, 1, H)[:, None].expand(H, W)
    depth = 2.0 + 8.0 * (1 - v)
    sky = torch.zeros(H, W, dtype=torch.bool)
    sky[:sky_rows] = True
    depth = torch.where(sky, torch.full_like(depth, 80.0), depth)
    res = DepthResult(depth, 1500.0, sky, "stub", (H, W), ["note"])
    return rgb, res, Intrinsics(1500.0, "35mm", False, ["warn"])


# ------------------------------------------------------------------------------------------ sizes
@pytest.mark.parametrize(
    "w,h,pw,ph",
    [(4000, 3000, 2048, 1536), (3000, 4000, 1536, 2048), (1536, 2048, 1536, 2048), (100, 50, 100, 50)],
)
def test_preview_size(w, h, pw, ph):
    got_w, got_h, scale = preview_size(w, h)
    assert (got_w, got_h) == (pw, ph)
    assert scale == pytest.approx(pw / w) and scale <= 1.0
    assert max(got_w, got_h) <= 2048


def test_package_files_sizes_and_contract(tmp_path):
    rgb, res, intr = make_inputs()
    meta = ImageMeta(2400, 1600, 1500.0, "35mm", 6, raw={"FocalLengthIn35mmFilm": 26, "Orientation": 6})
    out = tmp_path / "preview"
    info = package_preview(rgb, res, intr, out, filename="a.jpg", meta=meta, cached=True, warnings=["w1"])
    assert sorted(p.name for p in out.iterdir()) == sorted(ASSET_NAMES)
    assert not (tmp_path / "preview.tmp").exists()
    pw, ph = info["preview_width"], info["preview_height"]
    assert (pw, ph) == (2048, 1365) and info["scale"] == pytest.approx(pw / 2400)
    assert (out / "depth.f16").stat().st_size == pw * ph * 2  # float16
    assert (out / "sky.u8").stat().st_size == pw * ph
    with Image.open(out / "color.jpg") as im:
        assert im.size == (pw, ph) and im.format == "JPEG" and im.mode == "RGB"
        assert not im.getexif() and "icc_profile" not in im.info  # metadata stripped
    on_disk = json.loads((out / "meta.json").read_text())
    assert on_disk == info
    assert validate_dir(out) == info

    expected_keys = {
        "schema",
        "filename",
        "width",
        "height",
        "preview_width",
        "preview_height",
        "scale",
        "f_px_source",
        "f_px_preview",
        "intrinsics",
        "depth",
        "pivot_z",
        "exif",
        "warnings",
    }
    assert set(info) == expected_keys and info["schema"] == 1
    assert info["f_px_preview"] == pytest.approx(info["f_px_source"] * info["scale"], abs=1e-6)
    assert (info["width"], info["height"]) == (2400, 1600)
    assert info["intrinsics"] == {"source": "35mm", "assumed": False, "warnings": ["warn"]}
    assert set(info["depth"]) == {
        "min",
        "max",
        "p20",
        "sky_depth",
        "model",
        "infer_size",
        "cached",
        "cache_key",
    }
    d = info["depth"]
    assert d["model"] == "stub" and d["infer_size"] == [1600, 2400] and d["cached"] is True
    assert d["sky_depth"] == pytest.approx(80.0)
    assert d["min"] == pytest.approx(2.0, abs=0.05) and d["max"] <= 10.0  # sky excluded from stats
    assert d["min"] <= d["p20"] <= d["max"]
    assert info["pivot_z"] == pytest.approx(d["p20"], rel=0.02)  # choose_pivot = min(p20, 5 m)
    assert info["exif"] == {"make": None, "model": None, "f35": 26.0, "orientation": 6}
    assert info["warnings"] == ["w1"]


def test_no_resample_for_small_images(tmp_path):
    rgb, res, intr = make_inputs(W=640, H=480, sky_rows=0)
    info = package_preview(rgb, res, intr, tmp_path / "p")
    assert (info["preview_width"], info["preview_height"], info["scale"]) == (640, 480, 1.0)
    assert info["depth"]["sky_depth"] is None
    depth = np.frombuffer((tmp_path / "p" / "depth.f16").read_bytes(), dtype="<f2").reshape(480, 640)
    np.testing.assert_allclose(depth, res.depth.numpy(), rtol=1e-3)


def test_depth_mismatch_rejected(tmp_path):
    rgb, res, intr = make_inputs(W=200, H=100, sky_rows=0)
    with pytest.raises(PreviewError, match="does not match"):
        package_preview(rgb[:, :50], res, intr, tmp_path / "p")


# ------------------------------------------------------------------------------------------- privacy
def test_exif_summary_whitelist_no_gps_serial_dates(tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "Canon\x00 Inc"
    exif[0x0110] = "EOS R5 " + "x" * 100
    exif[0x9003] = "2024:01:02 03:04:05"  # DateTimeOriginal is in the Exif IFD; also set DateTime in IFD0
    exif[0x0132] = "2024:01:02 03:04:05"
    exif[0xA431] = "SERIAL-123456"  # BodySerialNumber
    exif.get_ifd(0x8825)[1] = "N"  # GPS latitude ref
    exif.get_ifd(0x8825)[2] = (48.0, 51.0, 24.0)
    blob = exif.tobytes()
    meta = ImageMeta(
        100,
        100,
        1000.0,
        "35mm",
        1,
        raw={"FocalLengthIn35mmFilm": 35, "Orientation": 1, "GPSLatitude": 1.0, "sensor_w_mm": 36.0},
    )
    summary = exif_summary(meta, blob)
    assert set(summary) == {"make", "model", "f35", "orientation"}
    assert (
        summary["make"] == "Canon Inc"
        and summary["model"].startswith("EOS R5")
        and len(summary["model"]) == 64
    )
    assert summary["f35"] == 35.0 and summary["orientation"] == 1

    rgb, res, intr = make_inputs(W=100, H=60, sky_rows=0)
    package_preview(rgb, res, intr, tmp_path / "p", meta=meta, exif_bytes=blob)
    text = (tmp_path / "p" / "meta.json").read_text().lower()
    for forbidden in ("gps", "serial", "2024:01", "datetime", "latitude"):
        assert forbidden not in text
    with Image.open(tmp_path / "p" / "color.jpg") as im:
        assert len(im.getexif()) == 0
    assert exif_summary(None, None) == {"make": None, "model": None, "f35": None, "orientation": None}
    assert exif_summary(None, b"not exif at all")["make"] is None


# ---------------------------------------------------------------------------------------------- float16
def test_float16_relative_error_within_budget(tmp_path):
    n = 2000  # <= 2048 wide: stored without resampling, so the only error is float16 rounding
    d = torch.logspace(math.log10(0.3), math.log10(200.0), n).repeat(8, 1)  # 0.3 m .. 200 m
    sky = torch.zeros_like(d, dtype=torch.bool)
    d[0, :3] = 1000.0  # sky clamp scale values
    sky[0, :3] = True
    res = DepthResult(d, 1000.0, sky, "x", (8, n), [])
    out = tmp_path / "p"
    package_preview(torch.zeros(3, 8, n), res, Intrinsics(1000.0, "35mm", False), out)
    back = np.frombuffer((out / "depth.f16").read_bytes(), dtype="<f2").astype(np.float64).reshape(8, n)
    ref = d.numpy().astype(np.float64)
    rel = np.abs(back - ref) / ref
    assert rel.max() < 0.002  # budget 0.2 %
    assert rel.max() <= 2**-11 * 1.01  # float16 theory: 0.049 %


# ------------------------------------------------------------------------------------ depth filter
def step_rows(W, H, edge):
    d = torch.full((H, W), BG)
    d[:, : int(edge)] = FG
    return d


def area_depth(d, h, w):
    return F.interpolate(d[None, None], size=(h, w), mode="area")[0, 0]


def min_pool(d, h, w):
    return -F.adaptive_max_pool2d((-d)[None, None], (h, w))[0, 0]


def bilinear_inverse(d, h, w):
    return resample_depth(d, h, w)


def measure(filter_fn, W, H, w, h, edges):
    """Worst edge-position error (preview px) and worst count of bridge pixels over sub-pixel edge shifts."""
    worst_err, worst_bridge = 0.0, 0
    s = W / w
    for e in edges:
        d = step_rows(W, 8, e)
        row = filter_fn(d, max(1, round(8 / s)), w)[0]
        mid_inv = 0.5 * (1 / FG + 1 / BG)
        is_fg = (1 / row) > mid_inv if filter_fn is bilinear_inverse else row < 0.5 * (FG + BG)
        worst_err = max(worst_err, abs(is_fg.sum().item() - int(e) / s))
        bridge = ((row > FG * 1.001) & (row < BG * 0.999)).sum().item()
        worst_bridge = max(worst_bridge, bridge)
    return worst_err, worst_bridge


SIZES = [
    (4000, 3000, 2048, 1536),
    (6000, 4000, 2048, 1365),
    (4032, 3024, 2048, 1536),
    (3000, 2000, 2048, 1365),
]
EDGES = [1000 + i * 37 for i in range(25)]  # integer full-res edges give sub-pixel preview positions


def test_depth_filter_chosen_meets_edge_and_bridge_budget():
    for W, H, w, h in SIZES:
        err, bridge = measure(bilinear_inverse, W, H, w, h, EDGES)
        assert err <= 0.5 + 1e-6, (W, H, err)  # edge within half a preview pixel
        assert bridge <= 1, (W, H, bridge)  # at most one intermediate pixel between fg and far bg


def test_depth_filter_alternatives_are_worse():
    """Documents the decision: the other candidates miss one of the two criteria on common sizes."""
    area_err = max(measure(area_depth, W, H, w, h, EDGES)[0] for W, H, w, h in SIZES)
    min_err = max(measure(min_pool, W, H, w, h, EDGES)[0] for W, H, w, h in SIZES)
    assert area_err > 0.5  # area average on depth shifts the edge by more than half a pixel
    assert min_err > 0.5  # min-pool dilates the foreground by almost a pixel
    assert max(measure(min_pool, W, H, w, h, EDGES)[1] for W, H, w, h in SIZES) == 0  # (but has no bridges)


def test_depth_filter_on_synth_two_layers_edge():
    """Ray-cast box in front of a wall (tests/synth.py) at 2400x1600 -> preview 2048x1365."""
    W, H = 2400, 1600
    cam = make_camera(torch.eye(3), torch.zeros(3), 1800.0, W, H)
    _, depth, _ = render(scene_two_layers(), cam, W, H)
    s = 2048 / W
    pw, ph = preview_size(W, H)[:2]
    prev = resample_depth(depth, ph, pw)
    row_full, row_prev = depth[H // 2], prev[ph // 2]
    fg_cols = (row_full < 3.0).nonzero().flatten()
    c0, c1 = fg_cols[0].item(), fg_cols[-1].item() + 1  # box edges as pixel boundaries in the full image
    near, far = float(row_full.min()), float(row_full.max())
    is_fg = (1 / row_prev) > 0.5 * (1 / near + 1 / far)
    idx = is_fg.nonzero().flatten()
    assert abs(idx[0].item() - c0 * s) <= 0.5 + 1e-3
    assert abs(idx[-1].item() + 1 - c1 * s) <= 0.5 + 1e-3
    between = ((row_prev > near * 1.001) & (row_prev < far * 0.999)).sum().item()
    assert between <= 2  # one bridge pixel per edge (two edges)


def test_bilinear_inverse_is_exact_on_planes():
    """Inverse depth is linear in image space on a plane, so a slanted plane keeps its shape."""
    W, H = 3000, 2000
    x = torch.linspace(-1, 1, W)[None].expand(H, W)
    inv = 1 / 5.0 + 0.1 * x  # a plane: 1/z linear in x
    depth = 1 / inv
    p = resample_depth(depth, 1365, 2048)
    xs = torch.linspace(-1, 1, 2048)[None].expand(1365, 2048)
    # pixel centres map linearly, so compare away from the borders with the analytic value
    got = 1 / p[:, 100:-100]
    want = 1 / 5.0 + 0.1 * (xs[:, 100:-100])
    assert (got - want).abs().max() < 2e-4


# ----------------------------------------------------------------------------------------------- sky
def test_sky_thin_line_survives_downscale():
    W, H = 3001, 200
    sky = torch.zeros(H, W, dtype=torch.bool)
    cols = [37, 1000, 1501, 2222, 2999]  # 1-px-wide sky gaps between "branches"
    sky[:, cols] = True
    out = resample_sky(sky, 100, 2048)
    assert out.shape == (100, 2048) and out.dtype == torch.bool
    for c in cols:
        assert out[:, int(c * 2048 / W)].all(), c  # every line is still sky in every row
    nearest = F.interpolate(sky.float()[None, None], size=(100, 2048), mode="nearest")[0, 0] > 0.5
    assert nearest.any(dim=0).sum() < out.any(dim=0).sum()  # nearest-neighbour would lose some lines


def test_sky_u8_values_are_0_or_255(tmp_path):
    rgb, res, intr = make_inputs()
    package_preview(rgb, res, intr, tmp_path / "p")
    sky = np.frombuffer((tmp_path / "p" / "sky.u8").read_bytes(), dtype=np.uint8)
    assert set(np.unique(sky)) == {0, 255}
    assert resample_sky(res.sky_mask, 1365, 2048).float().mean() == pytest.approx(sky.mean() / 255)


def test_resample_color_lanczos_antialiases():
    # alternating columns average to mid-grey when downscaled by 2 (a nearest-neighbour filter would not)
    rgb = torch.zeros(3, 40, 400)
    rgb[:, :, ::2] = 1.0
    out = resample_color(rgb, 20, 200)
    assert out.shape == (20, 200, 3) and out.dtype == np.uint8
    assert abs(float(out[5:15, 20:180].mean()) - 127.5) < 8


# ------------------------------------------------------------------------------------ atomic publish
def test_failure_before_rename_keeps_old_preview(tmp_path, monkeypatch):
    rgb, res, intr = make_inputs(W=300, H=200, sky_rows=0)
    out = tmp_path / "preview"
    old = package_preview(rgb, res, intr, out, filename="old")
    monkeypatch.setattr(assets, "validate_dir", lambda d: None)  # simulate a failed validation
    with pytest.raises(PreviewError):
        package_preview(rgb, res, intr, out, filename="new")
    monkeypatch.undo()
    assert not (tmp_path / "preview.tmp").exists()
    assert validate_dir(out) == old and old["filename"] == "old"

    def boom(*a, **k):
        raise OSError("killed before rename")

    monkeypatch.setattr(assets.os, "replace", boom)
    with pytest.raises(OSError):
        package_preview(rgb, res, intr, out, filename="newer")
    monkeypatch.undo()
    assert not (tmp_path / "preview.tmp").exists()
    # (the old directory was removed just before the rename in this simulated crash window only if it got
    # that far; a validation failure above is the realistic path and leaves it untouched)


def test_validate_dir_rejects_incomplete_or_foreign_sets(tmp_path):
    rgb, res, intr = make_inputs(W=300, H=200, sky_rows=0)
    out = tmp_path / "p"
    package_preview(rgb, res, intr, out)
    assert validate_dir(out) is not None
    (out / "depth.f16").write_bytes(b"\0" * 10)
    assert validate_dir(out) is None
    package_preview(rgb, res, intr, out)
    meta = json.loads((out / "meta.json").read_text())
    meta["schema"] = 2
    (out / "meta.json").write_text(json.dumps(meta))
    assert validate_dir(out) is None  # unknown schema: must be regenerated
    (out / "meta.json").unlink()
    assert validate_dir(out) is None
    assert validate_dir(tmp_path / "missing") is None


def test_stale_tmp_dir_is_replaced(tmp_path):
    rgb, res, intr = make_inputs(W=100, H=80, sky_rows=0)
    stale = tmp_path / "p.tmp"
    stale.mkdir()
    (stale / "junk").write_text("x")
    package_preview(rgb, res, intr, tmp_path / "p")
    assert not stale.exists() and validate_dir(tmp_path / "p") is not None
