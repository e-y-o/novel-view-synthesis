import struct
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from nvs.core import imageio
from nvs.core.imageio import load_image, save_image

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def _exif(orientation=None, focal35=None, gps=False, make=None):
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    if make:
        exif[0x010F] = make
    sub = exif.get_ifd(0x8769)
    if focal35:
        sub[0xA405] = focal35
    sub[0xA002], sub[0xA003] = 999, 888  # deliberately stale pixel dimensions
    exif[0x8769] = sub
    if gps:
        exif[0x8825] = {1: "N", 2: (48.0, 51.0, 24.0), 3: "E", 4: (2.0, 21.0, 7.0)}
    return exif


def _marker_image(w=64, h=32):
    """Asymmetric pattern: red top-left marker, blue bottom-right marker, grey elsewhere."""
    a = np.full((h, w, 3), 128, dtype=np.uint8)
    a[:8, :8] = (255, 0, 0)
    a[-8:, -8:] = (0, 0, 255)
    return a


# ---------- orientation ----------
def test_orientation_6_swaps_size_and_content(tmp_path):
    # Stored 64x32 with Orientation 6 (rotate 90 CW to display): displayed 32x64, red marker top-right.
    p = tmp_path / "o6.jpg"
    Image.fromarray(_marker_image()).save(p, quality=100, subsampling=0, exif=_exif(orientation=6))
    img = load_image(p)
    assert img.rgb.shape == (3, 64, 32) and (img.meta.width, img.meta.height) == (32, 64)
    px = (img.rgb * 255).round()
    assert px[:, 2, 29].tolist() == pytest.approx([255, 0, 0], abs=12)  # red now top-right
    assert px[:, 61, 2].tolist() == pytest.approx([0, 0, 255], abs=12)  # blue now bottom-left
    assert len(img.sha256) == 64


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orient_array_matches_pil(orientation):
    a = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    im = Image.fromarray(a)
    exif = Image.Exif()
    exif[0x0112] = orientation
    im.info["exif"] = exif.tobytes()
    want = np.asarray(ImageOps.exif_transpose(im))
    got = imageio._orient_array(a, orientation)
    assert np.array_equal(np.ascontiguousarray(got), want)


# ---------- colour management ----------
def _icc_profile(primaries, gamma, desc):
    """Minimal ICC v2 matrix/TRC display profile (primaries are D50-adapted XYZ columns)."""

    def s15(x):
        return struct.pack(">i", round(x * 65536))

    def xyz(v):
        return b"XYZ \x00\x00\x00\x00" + b"".join(s15(c) for c in v)

    ascii_desc = desc.encode() + b"\x00"
    desc_tag = b"desc\x00\x00\x00\x00" + struct.pack(">I", len(ascii_desc)) + ascii_desc + b"\x00" * 78
    curv = (
        b"curv\x00\x00\x00\x00" + struct.pack(">I", 1) + struct.pack(">H", round(gamma * 256)) + b"\x00\x00"
    )
    text = b"text\x00\x00\x00\x00" + b"public domain\x00"
    tags = {
        b"desc": desc_tag,
        b"cprt": text,
        b"wtpt": xyz((0.9642, 1.0, 0.8249)),
        b"rXYZ": xyz(primaries[0]),
        b"gXYZ": xyz(primaries[1]),
        b"bXYZ": xyz(primaries[2]),
        b"rTRC": curv,
        b"gTRC": curv,
        b"bTRC": curv,
    }
    pad = lambda b: b + b"\x00" * (-len(b) % 4)  # noqa: E731
    offset = 128 + 4 + 12 * len(tags)
    table, body = b"", b""
    for sig, blob in tags.items():
        table += sig + struct.pack(">II", offset + len(body), len(blob))
        body += pad(blob)
    header = (
        struct.pack(">I", offset + len(body))
        + b"\x00\x00\x00\x00"
        + b"\x02\x40\x00\x00"
        + b"mntrRGB XYZ "
        + b"\x00" * 12
        + b"acsp"
        + b"\x00" * 24
        + struct.pack(">I", 0)  # perceptual
        + s15(0.9642)
        + s15(1.0)
        + s15(0.8249)
        + b"\x00" * 48
    )
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + body


P3_D50 = ((0.5151, 0.2412, -0.0011), (0.2920, 0.6922, 0.0419), (0.1571, 0.0666, 0.7841))
SRGB_D50 = np.array([[0.4361, 0.3851, 0.1431], [0.2225, 0.7169, 0.0606], [0.0139, 0.0971, 0.7141]])


def _expected_srgb(p3_rgb, gamma=2.2):
    lin = np.asarray(p3_rgb) ** gamma
    xyz = np.array(P3_D50).T @ lin
    s = np.clip(np.linalg.inv(SRGB_D50) @ xyz, 0, 1)
    enc = np.where(s <= 0.0031308, 12.92 * s, 1.055 * s ** (1 / 2.4) - 0.055)
    return enc * 255


def test_display_p3_converted_to_srgb(tmp_path):
    p3 = (0.6, 0.5, 0.35)
    value = tuple(round(c * 255) for c in p3)
    p = tmp_path / "p3.png"
    Image.new("RGB", (16, 16), value).save(p, icc_profile=_icc_profile(P3_D50, 2.2, "Display P3 test"))
    img = load_image(p)
    got = (img.rgb[:, 8, 8] * 255).tolist()
    want = _expected_srgb([v / 255 for v in value])
    assert got == pytest.approx(want.tolist(), abs=2.0)
    assert any("Display P3 test" in n for n in img.notes)
    assert img.icc_profile is not None


def test_srgb_profile_left_untouched(tmp_path):
    from PIL import ImageCms

    p = tmp_path / "s.png"
    prof = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (8, 8), (10, 120, 200)).save(p, icc_profile=prof)
    img = load_image(p)
    assert (img.rgb[:, 0, 0] * 255).round().tolist() == [10, 120, 200]
    assert img.notes == []


# ---------- round trips ----------
def test_png_roundtrip_lossless(tmp_path):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (24, 40, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    a = load_image(tmp_path / "a.png")
    save_image(tmp_path / "b.png", a.rgb)
    b = load_image(tmp_path / "b.png")
    assert torch.equal(a.rgb, b.rgb)


def test_alpha_png_roundtrip_and_jpeg_warning(tmp_path):
    rgb = torch.rand(3, 10, 12, generator=torch.Generator().manual_seed(1))
    alpha = torch.rand(10, 12, generator=torch.Generator().manual_seed(2))
    save_image(tmp_path / "x.png", rgb, alpha=alpha)
    out = Image.open(tmp_path / "x.png")
    assert out.mode == "RGBA"
    a = np.asarray(out)
    assert np.array_equal(a[:, :, 3], (alpha * 255).round().byte().numpy())
    with pytest.warns(UserWarning, match="alpha"):
        save_image(tmp_path / "x.jpg", rgb, alpha=alpha)
    assert Image.open(tmp_path / "x.jpg").mode == "RGB"


def _png16(arr, ctype):
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].astype(">u2").tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, ctype, 0, 0, 0)
    idat = zlib.compress(raw)
    return imageio._PNG_SIG + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_16bit_rgb_png_precision(tmp_path):
    rng = np.random.default_rng(3)
    a = rng.integers(0, 65536, (9, 11, 3), dtype=np.uint16)
    (tmp_path / "s.png").write_bytes(_png16(a, 2))
    img = load_image(tmp_path / "s.png")
    want = torch.from_numpy(a.astype(np.float32) / 65535).permute(2, 0, 1)
    assert (img.rgb - want).abs().max() <= 1 / 65535 + 1e-7


def test_16bit_png_filters_and_grey(tmp_path):
    # Real encoders choose Sub/Up/Average/Paeth filters: Pillow writes 16-bit grey with adaptive filtering.
    rng = np.random.default_rng(4)
    g = rng.integers(0, 65536, (20, 30), dtype=np.uint16)
    Image.fromarray(g).save(tmp_path / "g.png")
    img = load_image(tmp_path / "g.png")
    want = torch.from_numpy(g.astype(np.float32) / 65535)
    assert (img.rgb[0] - want).abs().max() <= 1 / 65535 + 1e-7
    assert torch.equal(img.rgb[0], img.rgb[2])
    # the pure-Python decoder agrees with Pillow on every filter type
    arr, _ = imageio._read_png16((tmp_path / "g.png").read_bytes())
    assert np.array_equal(arr[:, :, 0], g)


def test_rgba_and_cmyk_and_gray_inputs(tmp_path):
    Image.new("RGBA", (6, 6), (255, 0, 0, 0)).save(tmp_path / "t.png")  # fully transparent red
    img = load_image(tmp_path / "t.png")
    assert (img.rgb == 1.0).all() and any("alpha" in n for n in img.notes)
    Image.new("L", (6, 6), 100).save(tmp_path / "l.png")
    img = load_image(tmp_path / "l.png")
    assert (img.rgb * 255).round().unique().tolist() == [100]
    Image.new("CMYK", (6, 6), (0, 0, 0, 0)).save(tmp_path / "c.jpg")
    img = load_image(tmp_path / "c.jpg")
    assert img.rgb.shape == (3, 6, 6)


def test_refuses_huge_images(tmp_path, monkeypatch):
    Image.new("RGB", (40, 30)).save(tmp_path / "h.png")
    monkeypatch.setattr(imageio, "MAX_PIXELS", 1000)
    with pytest.raises(ValueError, match="exceeds"):
        load_image(tmp_path / "h.png")


# ---------- EXIF on save ----------
def test_saved_jpeg_exif(tmp_path):
    p = tmp_path / "src.jpg"
    Image.fromarray(_marker_image()).save(p, exif=_exif(orientation=6, focal35=26, gps=True, make="TestCam"))
    src = load_image(p)
    assert src.exif_bytes is not None
    out = tmp_path / "out.jpg"
    save_image(out, src.rgb, source=src, comment="Novel view synthesis, yaw=5")
    ex = Image.open(out).getexif()
    assert ex.get(0x0112) == 1
    assert ex.get(0x010F) == "TestCam"
    sub = ex.get_ifd(0x8769)
    assert sub[0xA405] == 26
    assert (sub[0xA002], sub[0xA003]) == (32, 64)  # oriented output size
    assert 0x8825 not in ex and not ex.get_ifd(0x8825)
    assert 0x0201 not in ex.get_ifd(1)  # no embedded thumbnail
    assert b"yaw=5" in sub[0x9286]
    assert (Image.open(out).size) == (32, 64)

    keep = tmp_path / "keep.jpg"
    save_image(keep, src.rgb, source=src, keep_gps=True)
    assert Image.open(keep).getexif().get_ifd(0x8825).get(1) == "N"


def test_saved_png_carries_exif_and_srgb_profile(tmp_path):
    p = tmp_path / "src.jpg"
    Image.fromarray(_marker_image()).save(p, exif=_exif(focal35=50))
    src = load_image(p)
    out = tmp_path / "o.png"
    save_image(out, src.rgb, source=src)
    im = Image.open(out)
    assert im.getexif().get_ifd(0x8769)[0xA405] == 50
    assert im.info.get("icc_profile")
    assert load_image(out).meta.f_px == pytest.approx(src.meta.f_px)


def test_save_validates(tmp_path):
    with pytest.raises(ValueError):
        save_image(tmp_path / "x.bmp", torch.zeros(3, 4, 4))
    with pytest.raises(ValueError):
        save_image(tmp_path / "x.png", torch.zeros(4, 4))
    # rounding, not truncation; clamping
    save_image(tmp_path / "r.png", torch.tensor([0.999, 0.5, -1.0]).reshape(3, 1, 1).expand(3, 2, 2))
    assert np.asarray(Image.open(tmp_path / "r.png"))[0, 0].tolist() == [255, 128, 0]


# ---------- real samples ----------
@pytest.mark.skipif(not SAMPLES.exists(), reason="data/samples not present")
def test_samples_load_and_save(tmp_path):
    files = [f for f in sorted(SAMPLES.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".heic")]
    assert files
    for f in files:
        img = load_image(f)
        assert (img.rgb.shape[2], img.rgb.shape[1]) == (img.meta.width, img.meta.height), f.name
        assert 0 <= float(img.rgb.min()) and float(img.rgb.max()) <= 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            save_image(tmp_path / (f.stem + ".jpg"), img.rgb, source=img)
        again = Image.open(tmp_path / (f.stem + ".jpg"))
        assert again.size == (img.meta.width, img.meta.height)
        if img.exif_bytes:
            assert again.getexif().get(0x0112) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not SAMPLES.exists(), reason="data/samples not present")
def test_load_speed_to_gpu():
    import time

    biggest = max(
        (f for f in SAMPLES.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda f: f.stat().st_size,
    )
    load_image(biggest, device="cuda")  # warm up
    t0 = time.perf_counter()
    img = load_image(biggest, device="cuda")
    dt = time.perf_counter() - t0
    print(f"load {biggest.name} {img.meta.width}x{img.meta.height} -> cuda: {dt:.3f}s")
    assert dt < 0.5
