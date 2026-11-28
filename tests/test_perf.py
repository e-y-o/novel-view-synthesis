"""Performance guards for the render budgets (this machine: RTX 5070 Ti; other GPUs: set NVS_PERF_FACTOR).

Limits are the design budgets and sit 1.3-2x above what was measured here (12 MP splat ~100-115 ms,
classify ~12 ms warm), so ordinary noise does not trip them but a real regression (e.g. losing the batched
kernel taps) does. ``NVS_PERF_FACTOR=2`` doubles every limit for a slower GPU.
"""

import os
import statistics
import time

import pytest
import torch

from nvs.core.camera import make_camera, orbit_pose
from nvs.render.holes import classify_holes
from nvs.render.splat import SplatOptions, splat

FACTOR = float(os.environ.get("NVS_PERF_FACTOR", "1"))
pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _median_ms(fn, runs=7):
    fn()
    fn()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def test_splat_and_classify_12mp_budget():
    w, h = 4000, 3000
    ys = torch.linspace(0, 1, h, device="cuda")[:, None].expand(h, w)
    xs = torch.linspace(0, 1, w, device="cuda")[None, :].expand(h, w)
    color = torch.stack([xs, ys, (xs + ys) / 2])
    depth = 2.0 + 6.0 * ys + 0.5 * torch.sin(xs * 30)
    depth[h // 3 : 2 * h // 3, w // 3 : w // 2] = 1.5
    R, t = orbit_pose(8, 0, 3.0)
    cam = make_camera(R, t, 3600.0, w, h)
    res = splat(color, depth, cam, options=SplatOptions(extra_channels=True))
    splat_ms = _median_ms(lambda: splat(color, depth, cam, options=SplatOptions(extra_channels=True)))
    classify_ms = _median_ms(lambda: classify_holes(res, depth, cam))
    print(f"\n12 MP: splat {splat_ms:.0f} ms (limit {150 * FACTOR:.0f}), classify {classify_ms:.0f} ms "
          f"(limit {60 * FACTOR:.0f})")  # fmt: skip
    assert splat_ms <= 150 * FACTOR
    assert classify_ms <= 60 * FACTOR
