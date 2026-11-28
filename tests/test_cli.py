import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from synth import render, scene_two_layers

from nvs.cli import render as cli_render
from nvs.cli.main import build_parser, main
from nvs.cli.render import CliError, run_render
from nvs.core.camera import make_camera

W, H, F = 256, 192, 250.0


@pytest.fixture
def photo(tmp_path):
    """Synthetic 'photo' (two-layer scene) with EXIF, plus its exact depth as a .npy file."""
    cam = make_camera(torch.eye(3), torch.zeros(3), F, W, H)
    color, depth, _ = render(scene_two_layers(), cam)
    exif = Image.Exif()
    exif[0x0112] = 1
    exif[0x010F] = "TestCam"
    sub = exif.get_ifd(0x8769)
    sub[0xA405] = 30  # FocalLengthIn35mmFilm
    exif[0x8769] = sub
    path = tmp_path / "photo.jpg"
    arr = (color.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path, quality=98, exif=exif)
    np.save(tmp_path / "depth.npy", depth.numpy())
    return path, tmp_path / "depth.npy"


def parse(*argv):
    return build_parser().parse_args(["render", *map(str, argv)])


def test_help_and_version_exit_zero(capsys):
    for flag in ("--help", "--version"):
        with pytest.raises(SystemExit) as e:
            main([flag])
        assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        main(["render", "--help"])
    assert e.value.code == 0
    assert "--yaw" in capsys.readouterr().out


def test_help_is_fast_and_does_not_import_torch():
    code = (
        "import sys, time; t=time.perf_counter(); from nvs.cli.main import build_parser; build_parser();"
        "assert 'torch' not in sys.modules, 'torch imported'; print(time.perf_counter()-t)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert float(out.stdout.strip()) < 1.0


def test_usage_error_exit_code_2():
    with pytest.raises(SystemExit) as e:
        main(["render"])  # missing image
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["render", "x.jpg", "--hole", "pink"])
    assert e.value.code == 2


def test_render_synthetic_cpu_rgba_png(photo, tmp_path):
    img, depth = photo
    out = tmp_path / "out.png"
    rep = run_render(parse(img, "--depth-file", depth, "--device", "cpu", "--yaw", 5, "--out", out))
    assert (rep.width, rep.height) == (W, H)
    im = Image.open(out)
    assert im.size == (W, H) and im.mode == "RGBA"
    a = np.asarray(im)[:, :, 3]
    assert a.min() == 0 and a.max() == 255  # holes transparent, the rest opaque
    ex = im.getexif()
    assert ex.get(0x0112) == 1 and ex.get(0x010F) == "TestCam"
    assert ex.get_ifd(0x8769)[0xA405] == 30
    assert b"nvs render yaw=5" in ex.get_ifd(0x8769)[0x9286]
    assert rep.intrinsics.source == "35mm" and rep.coverage["total"] > 0
    assert rep.depth_model == "depth-file" and set(rep.timings) >= {"load", "depth", "splat", "save"}


def test_hole_colours_and_extra_outputs(photo, tmp_path):
    img, depth = photo
    out = tmp_path / "m.png"
    mask, classes, vis = tmp_path / "mask.png", tmp_path / "classes.png", tmp_path / "vis.png"
    rep = run_render(
        parse(
            img,
            "--depth-file",
            depth,
            "--device",
            "cpu",
            "--yaw",
            6,
            "--hole",
            "magenta",
            "--out",
            out,
            "--save-mask",
            mask,
            "--save-classes",
            classes,
            "--save-depth-vis",
            vis,
        )
    )
    arr = np.asarray(Image.open(out))
    assert arr.shape == (H, W, 3)
    assert ((arr == [255, 0, 255]).all(-1)).any()
    assert (
        Image.open(mask).mode == "L" and Image.open(classes).size == (W, H) and Image.open(vis).size == (W, H)
    )
    assert rep.hole_mode == "magenta"


def test_jpeg_with_alpha_falls_back_to_magenta(photo, tmp_path):
    img, depth = photo
    rep = run_render(
        parse(img, "--depth-file", depth, "--device", "cpu", "--yaw", 6, "--out", tmp_path / "o.jpg")
    )
    assert rep.hole_mode == "magenta" and any("magenta" in w for w in rep.warnings)
    assert Image.open(tmp_path / "o.jpg").size == (W, H)


def test_f_px_override(photo, tmp_path):
    img, depth = photo
    rep = run_render(
        parse(img, "--depth-file", depth, "--device", "cpu", "--f-px", 321.5, "--out", tmp_path / "o.png")
    )
    assert rep.intrinsics.source == "override" and rep.intrinsics.f_px == 321.5
    rep = run_render(
        parse(img, "--depth-file", depth, "--device", "cpu", "--focal-35mm", 50, "--out", tmp_path / "o.png")
    )
    assert rep.intrinsics.f_px == pytest.approx(50 * (W**2 + H**2) ** 0.5 / 43.267)


def test_clamp_limits_a_large_move(photo, tmp_path):
    img, depth = photo
    out = tmp_path / "o.png"
    free = run_render(parse(img, "--depth-file", depth, "--device", "cpu", "--yaw", 25, "--out", out))
    lim = run_render(
        parse(img, "--depth-file", depth, "--device", "cpu", "--yaw", 25, "--clamp", "--out", out)
    )
    assert any("about" in w and "holes" in w for w in free.warnings)  # unclamped: warns
    assert lim.scale < 1 and lim.pose["yaw"] < 25 and any("limited" in w for w in lim.warnings)
    assert lim.coverage["total"] < free.coverage["total"]


def test_missing_file_message_and_exit_code(tmp_path, capsys):
    assert main(["render", str(tmp_path / "nope.jpg"), "--device", "cpu"]) == 1
    assert "file not found" in capsys.readouterr().err


def test_depth_file_shape_mismatch(photo, tmp_path, capsys):
    img, _ = photo
    bad = tmp_path / "bad.npy"
    np.save(bad, np.ones((10, 10), np.float32))
    assert (
        main(
            [
                "render",
                str(img),
                "--depth-file",
                str(bad),
                "--device",
                "cpu",
                "--out",
                str(tmp_path / "o.png"),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "256x192" in err and "10x10" in err and "Traceback" not in err


def test_not_an_image_and_bad_output_format(tmp_path, photo, capsys):
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"not an image")
    _, depth = photo
    assert main(["render", str(junk), "--depth-file", str(depth), "--device", "cpu"]) == 1
    assert "cannot read" in capsys.readouterr().err
    img, _ = photo
    assert (
        main(
            [
                "render",
                str(img),
                "--depth-file",
                str(depth),
                "--device",
                "cpu",
                "--out",
                str(tmp_path / "o.bmp"),
            ]
        )
        == 1
    )
    assert "unsupported output format" in capsys.readouterr().err


def test_cpu_without_depth_file_is_explained(photo, capsys):
    img, _ = photo
    assert main(["render", str(img), "--device", "cpu"]) == 1
    assert "--depth-file" in capsys.readouterr().err


def test_license_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("NVS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(cli_render.ACCEPT_ENV, raising=False)
    args = parse("x.jpg")
    monkeypatch.setattr(sys, "stdin", None)  # non-interactive
    with pytest.raises(CliError, match="--accept-license"):
        cli_render.ensure_license(args)
    monkeypatch.setenv(cli_render.ACCEPT_ENV, "1")
    cli_render.ensure_license(args)
    assert (tmp_path / "depthpro2-license-accepted").exists()
    monkeypatch.delenv(cli_render.ACCEPT_ENV)
    cli_render.ensure_license(args)  # remembered
