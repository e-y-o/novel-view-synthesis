import time

import pytest
import torch
import torch.nn.functional as F

from nvs.depth.base import RawDepth
from nvs.depth.refine import refine_depth, sky_and_clamp
from nvs.depth.visualize import depth_to_color

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, H = 512, 384
EDGE = 203  # true depth/colour edge: columns < EDGE are near (2 m), >= EDGE far (4 m)


def _step_scene():
    depth = torch.full((H, W), 2.0)
    depth[:, EDGE:] = 4.0
    rgb = torch.zeros(3, H, W)
    rgb[0, :, :EDGE] = 0.9
    rgb[2, :, EDGE:] = 0.9
    return depth.to(DEV), rgb.to(DEV)


def _raw(depth_lr, scale, f=500.0):
    h, w = depth_lr.shape
    return RawDepth(depth_lr, f, (h, w), scale)


def _transition(row: torch.Tensor):
    """(first index where depth crosses the midpoint, number of pixels strictly between 10 % and 90 %)."""
    lo, hi = float(row.min()), float(row.max())
    mid = (row > (lo + hi) / 2).nonzero()[0].item()
    width = int(((row > lo + 0.1 * (hi - lo)) & (row < lo + 0.9 * (hi - lo))).sum())
    return mid, width


def test_sky_becomes_finite_and_masked():
    depth = torch.full((40, 50), 3.0)
    depth += torch.linspace(0, 2, 50)[None]
    depth[:10] = float("inf")
    depth[10:14] = 1e5
    out, sky, _ = sky_and_clamp(depth)
    assert torch.isfinite(out).all()
    assert sky[:14].all() and not sky[20:].any()
    assert float(out[:14].min()) == pytest.approx(float(out[:14].max()))  # sky at one constant depth
    assert float(out[:14].max()) < 100


def test_normal_depth_untouched():
    g = torch.Generator().manual_seed(0)
    depth = 2.0 + 2.0 * torch.rand(60, 80, generator=g)
    out, sky, _ = sky_and_clamp(depth)
    assert torch.equal(out, depth) and not sky.any()


def test_near_zero_spike_clamped():
    g = torch.Generator().manual_seed(1)
    depth = 2.0 + 2.0 * torch.rand(100, 100, generator=g)
    depth[50, 50] = 1e-4
    out, _, _ = sky_and_clamp(depth)
    assert out[50, 50] > 1.5


def test_all_sky_input():
    out, sky, notes = sky_and_clamp(torch.full((8, 8), float("inf")))
    assert torch.isfinite(out).all() and sky.all() and notes


def test_guided_upsample_keeps_edge():
    depth, rgb = _step_scene()
    lr = depth[::4, ::4].contiguous()  # nearest 4x downsample: the edge lands on a 4 px grid
    res = refine_depth(_raw(lr, 0.25), rgb)
    row = res.depth[H // 2].cpu()
    mid, width = _transition(row)
    assert abs(mid - EDGE) <= 1
    assert width <= 2
    bilinear = F.interpolate(lr[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    b_mid, b_width = _transition(bilinear[H // 2].cpu())
    assert width < b_width
    assert (res.depth.min().item(), res.depth.max().item()) == pytest.approx((2.0, 4.0), abs=0.15)


def test_flat_rgb_matches_bilinear():
    xs = torch.linspace(0, 1, W)
    depth = (2.0 + 2.0 * xs)[None].expand(H, W).contiguous().to(DEV)
    rgb = torch.full((3, H, W), 0.5, device=DEV)
    lr = F.interpolate(depth[None, None], size=(H // 4, W // 4), mode="area")[0, 0]
    res = refine_depth(_raw(lr, 0.25), rgb)
    inv = F.interpolate((1 / lr)[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    assert (res.depth - 1 / inv).abs().mean() < 0.02
    # a step in depth with a flat photo must not become sharper than a smooth upsample
    step = torch.full((H // 4, W // 4), 2.0, device=DEV)
    step[:, W // 8 :] = 4.0
    res = refine_depth(_raw(step, 0.25), rgb)
    assert _transition(res.depth[H // 2].cpu())[1] >= 3


def test_output_contract():
    depth, rgb = _step_scene()
    lr = depth[::2, ::2].contiguous()
    res = refine_depth(_raw(lr, 0.5, f=400.0), rgb, model="m")
    assert res.depth.shape == (H, W) and res.depth.dtype == torch.float32 and res.depth.device == rgb.device
    assert res.sky_mask.shape == (H, W) and res.sky_mask.dtype == torch.bool
    assert res.f_px == pytest.approx(800.0)  # scale 0.5 -> x2
    assert res.model == "m" and res.infer_size == (H // 2, W // 2)
    assert torch.isfinite(res.depth).all()


def test_no_upsample_when_sizes_match():
    depth, rgb = _step_scene()
    res = refine_depth(_raw(depth, 1.0), rgb)
    assert torch.equal(res.depth, depth) and res.f_px == 500.0


def test_sky_upsampled_with_guide():
    depth, rgb = _step_scene()
    depth = depth.clone()
    depth[:, EDGE:] = float("inf")
    res = refine_depth(_raw(depth[::4, ::4].contiguous(), 0.25), rgb)
    assert torch.isfinite(res.depth).all()
    cols = res.sky_mask[H // 2].nonzero().flatten()
    assert abs(int(cols.min()) - EDGE) <= 1 and int(cols.max()) == W - 1


def test_depth_to_color():
    depth = torch.linspace(1, 10, 100 * 80).reshape(80, 100)
    sky = torch.zeros(80, 100, dtype=torch.bool)
    sky[:5] = True
    rgb = depth_to_color(depth, sky)
    assert rgb.shape == (3, 80, 100) and 0 <= float(rgb.min()) and float(rgb.max()) <= 1
    assert rgb[:, :5].sum() == 0  # sky is black
    assert not torch.equal(rgb[:, 10, 0], rgb[:, 79, 99])  # near and far differ


@pytest.mark.gpu
def test_24mp_speed_and_memory():
    g = torch.Generator(device="cuda").manual_seed(0)
    h, w = 4284, 5712
    rgb = torch.rand(3, h, w, device="cuda", generator=g)
    lr = 2 + 3 * torch.rand(1414, 1887, device="cuda", generator=g)
    lr[:100] = float("inf")
    raw = _raw(lr, 1887 / 5712)
    refine_depth(raw, rgb)  # warm-up
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = refine_depth(raw, rgb)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    extra = (torch.cuda.max_memory_allocated() - base) / 2**30
    print(f"\n24 MP refine: {dt:.2f} s, extra peak {extra:.2f} GiB")
    assert res.depth.shape == (h, w) and torch.isfinite(res.depth).all()
    assert dt < 1.0 and extra < 3.0
