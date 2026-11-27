import time

import pytest
import torch

from nvs.depth.base import DepthBackend, DepthError, RawDepth, plan_inference


class FakeBackend:
    name = "fake"

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def estimate_focal(self, rgb):
        return 1000.0

    def infer(self, rgb, f_px):
        _, h, w = rgb.shape
        return RawDepth(torch.ones(h, w), f_px or 1000.0, (h, w), 1.0)


def test_fake_satisfies_protocol():
    assert isinstance(FakeBackend(), DepthBackend)


def test_plan_downscales_large_images():
    p = plan_inference(4000, 3000, 3000.0, 4.0, 6.0)
    assert p.width * p.height <= 4e6 and p.scale < 1
    assert p.f_px == pytest.approx(3000.0 * p.width / 4000)
    assert abs(p.width / p.height - 4 / 3) < 0.01


def test_plan_leaves_small_images_alone():
    p = plan_inference(1000, 800, 900.0, 4.0, 6.0)
    assert (p.width, p.height, p.scale, p.f_px) == (1000, 800, 1.0, 900.0)
    assert plan_inference(2000, 2000, 900.0, 4.0, 6.0).scale == 1.0  # exactly 4 MP


def test_plan_without_focal_and_invalid():
    assert plan_inference(6000, 4000, None, 4.0, 6.0).f_px is None
    with pytest.raises(ValueError):
        plan_inference(1000, 800, 900.0, 7.0, 6.0)  # above the hard max
    with pytest.raises(ValueError):
        plan_inference(0, 800, 900.0, 4.0, 6.0)
    with pytest.raises(ValueError):
        plan_inference(100, 100, 900.0, 0.0, 6.0)


@pytest.mark.parametrize("wh", [(24_000, 1000), (4284, 5712), (1, 5_000_000), (4001, 1001)])
def test_plan_never_exceeds_limit(wh):
    p = plan_inference(*wh, 100.0, 4.0, 6.0)
    assert p.width * p.height <= 4e6


def test_backend_constructor_validation():
    from nvs.depth.depthpro2 import DepthPro2Backend

    with pytest.raises(ValueError):
        DepthPro2Backend(max_mp=8.0, hard_max_mp=6.0)
    with pytest.raises(ValueError):
        DepthPro2Backend(precision="int4")
    with pytest.raises(ValueError):
        DepthPro2Backend(vram_fraction=1.5)


def test_depth_error_is_runtime_error():
    assert issubclass(DepthError, RuntimeError)


def _scene_rgb(w, h):
    """Deterministic structured image (gradient + blocks + noise); stands in for a photo."""
    g = torch.Generator().manual_seed(0)
    y, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    img = torch.stack([x, y, (x + y) / 2])
    img[:, h // 3 : 2 * h // 3, w // 3 : w // 2] = 0.9
    return (img + 0.05 * torch.rand(3, h, w, generator=g)).clamp(0, 1)


@pytest.mark.gpu
@pytest.mark.slow
def test_depthpro2_h_end_to_end():
    from nvs.depth.depthpro2 import DepthPro2Backend

    be = DepthPro2Backend()
    base = torch.cuda.memory_allocated()
    be.load()
    loaded = torch.cuda.memory_allocated()
    assert loaded - base > 3 * 2**30
    rgb = _scene_rgb(1280, 800)
    raw = be.infer(rgb, f_px=1100.0)
    assert raw.depth.shape == (800, 1280) and raw.scale == 1.0 and raw.f_px == 1100.0
    assert not torch.isnan(raw.depth).any() and (raw.depth[torch.isfinite(raw.depth)] > 0).all()
    est = be.estimate_focal(rgb)
    assert est is not None and est > 0 and be.estimate_focal(rgb) == est  # cached
    raw2 = be.infer(rgb, f_px=None)
    assert raw2.f_px > 0 and any("estimated" in n for n in raw2.notes)
    be.unload()
    assert loaded - torch.cuda.memory_allocated() > 3 * 2**30


@pytest.mark.gpu
@pytest.mark.slow
def test_depthpro2_h_4mp_timing_and_peak_vram():
    from nvs.depth.depthpro2 import DepthPro2Backend

    be = DepthPro2Backend()
    be.load()
    rgb = _scene_rgb(2828, 1414)  # exactly ~4 MP
    be.infer(rgb, f_px=2500.0)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    raw = be.infer(rgb, f_px=2500.0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\n4 MP H+ bf16: {dt:.2f} s, peak {peak:.2f} GiB, infer_size={raw.infer_size}")
    assert raw.scale == 1.0
    assert dt <= 1.5 * 1.2 and peak <= 10.5
    # a 12 MP request is shrunk to <= 4 MP and the focal length scaled with it
    big = _scene_rgb(4000, 3000)
    raw = be.infer(big, f_px=3600.0)
    assert raw.infer_size[0] * raw.infer_size[1] <= 4e6
    assert raw.f_px == pytest.approx(3600.0 * raw.scale)
    be.unload()
