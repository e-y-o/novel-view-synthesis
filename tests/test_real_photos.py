"""Smoke test on the local sample photos with the real depth model (skipped when data/samples is absent).

Uses the depth cache (data/cache), so after the first run it only exercises load, splat, classify and save;
the first run per photo also loads Depth Pro 2 (about 6 s).
"""

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nvs.cli.main import build_parser
from nvs.cli.render import run_render
from nvs.core.imageio import load_image

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
PHOTOS = (
    sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".heic"))
    if SAMPLES.exists()
    else []
)
# At yaw 5 deg a telephoto photo can lose ~30 % of its frame (out of frame), so the ceiling is 35 %, not 25 %.
MAX_HOLE_FRACTION = 0.35
MIN_HOLE_FRACTION = 0.001

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.skipif(not PHOTOS, reason="no sample photos in data/samples")
@pytest.mark.parametrize("photo", PHOTOS, ids=lambda p: p.name)
def test_sample_photo_renders(photo, tmp_path, monkeypatch):
    monkeypatch.setenv("NVS_ACCEPT_DEPTHPRO2_LICENSE", "1")
    out = tmp_path / "out.png"
    args = build_parser().parse_args(["render", str(photo), "--yaw", "5", "--out", str(out)])
    t0 = time.perf_counter()
    report = run_render(args)
    elapsed = time.perf_counter() - t0
    src = load_image(photo)
    result = np.asarray(Image.open(out))
    assert result.shape[:2] == (src.meta.height, src.meta.width)
    assert (report.width, report.height) == (src.meta.width, src.meta.height)
    assert np.isfinite(result.astype(np.float32)).all()
    assert MIN_HOLE_FRACTION <= report.coverage["total"] <= MAX_HOLE_FRACTION
    assert not math.isnan(report.intrinsics.f_px)
    assert elapsed < 10, f"{photo.name} took {elapsed:.1f} s"
    torch.cuda.empty_cache()
