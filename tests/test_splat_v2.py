import statistics
import time

import pytest
import torch
import torch.nn.functional as F
from synth import psnr, render, scene_room, scene_two_layers, visible_in_both

from nvs.core.camera import make_camera, orbit_pose
from nvs.render.splat import SplatOptions, splat, splat_legacy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, H, F_PX = 512, 384, 500.0
SRC = make_camera(torch.eye(3), torch.zeros(3), F_PX, W, H)


def cam_shift(dx):
    return make_camera(torch.eye(3), torch.tensor([dx, 0.0, 0.0]), F_PX, W, H)


def centroid_x(weight):
    cols = torch.arange(weight.shape[1], dtype=torch.float32, device=weight.device)
    return float((weight.sum(0) * cols).sum() / weight.sum())


def build_plane_square():
    color = torch.full((3, H, W), 0.5, device=DEV)
    depth = torch.full((H, W), 3.0, device=DEV)
    color[:, :, 100:140] = torch.tensor([1.0, 0, 0], device=DEV)[:, None, None]
    color[:, 140:260, 200:300] = torch.tensor([0, 0, 1.0], device=DEV)[:, None, None]
    depth[140:260, 200:300] = 1.0
    return color, depth


def scene_source(scene, device=DEV):
    c, d, _ = render(scene, SRC)
    return c.to(device), d.to(device)


def isolated_holes(hole: torch.Tensor) -> int:
    """Hole pixels whose 9x9 neighbourhood is < 30 % hole (8 px border excluded: slivers there are real)."""
    dens = F.avg_pool2d(hole.float()[None, None], 9, 1, 4, count_include_pad=False)[0, 0]
    iso = hole & (dens < 0.30)
    return int(iso[8:-8, 8:-8].sum())


def test_plane_and_square_shift_like_m0():
    color, depth = build_plane_square()
    res = splat(color, depth, cam_shift(0.1))
    img, hole = res.image, res.hole
    red = ((img[0] > 0.8) & (img[1] < 0.2) & (img[2] < 0.2)).float()
    blue = ((img[2] > 0.8) & (img[0] < 0.2)).float()
    assert abs(centroid_x(red) - (119.5 - F_PX * 0.1 / 3)) <= 0.5
    assert abs(centroid_x(blue) - (249.5 - F_PX * 0.1 / 1)) <= 0.5
    row = slice(180, 220)
    new_right = 299.5 - F_PX * 0.1
    assert hole[row, int(new_right) + 6 : int(new_right) + 28].float().mean() > 0.95
    assert hole[row, int(new_right) - 130 : int(new_right) - 110].float().mean() < 0.01


def test_result_contract_and_extra():
    color, depth = build_plane_square()
    extra = torch.ones(2, H, W, device=DEV)
    extra[1] = 0.25
    res = splat(color, depth, cam_shift(0.1), extra=extra)
    assert res.image.shape == (3, H, W) and res.alpha.shape == (H, W) and res.zbuf.shape == (H, W)
    assert res.extra.shape == (2, H, W)
    assert torch.allclose(res.extra[0][~res.hole], torch.ones(1, device=DEV), atol=1e-5)
    assert torch.allclose(res.extra[1][~res.hole], torch.full((1,), 0.25, device=DEV), atol=1e-5)
    assert (res.extra[:, res.hole] == 0).all()
    assert torch.isinf(res.zbuf[res.hole]).all() and torch.isfinite(res.zbuf[~res.hole]).all()
    assert (res.alpha[res.hole] == 0).all() and float(res.alpha.max()) == pytest.approx(1.0)
    assert 0 <= float(res.alpha.min()) and float(res.alpha.max()) <= 1
    assert splat(color, depth, cam_shift(0.1)).extra is None


def test_size_mismatch_rejected():
    color, depth = build_plane_square()
    with pytest.raises(ValueError):
        splat(color, depth, make_camera(torch.eye(3), torch.zeros(3), F_PX, 100, 100))
    with pytest.raises(ValueError):
        splat(color, depth[:10], SRC)


def test_identity_reproduces_source():
    color, depth = scene_source(scene_two_layers())
    res = splat(color, depth, SRC)
    assert not res.hole.any()
    assert (res.image - color).abs().max() < 2e-3


@pytest.mark.parametrize("yaw", [15.0, 25.0])
def test_isolated_holes_drop_vs_legacy(yaw):
    color, depth = scene_source(scene_room())
    R, t = orbit_pose(yaw, 0, 2.5)
    cam = make_camera(R, t, F_PX, W, H)
    new = splat(color, depth, cam)
    _, old_hole = splat_legacy(color, depth, F_PX, R, t)
    n_old, n_new = isolated_holes(old_hole), isolated_holes(new.hole)
    print(f"\nyaw {yaw}: isolated hole pixels legacy {n_old} -> v2 {n_new}")
    assert n_old > 100
    assert n_new <= 0.1 * n_old


def test_foreground_silhouette_does_not_grow():
    color, depth = scene_source(scene_two_layers())
    cam = cam_shift(0.2)
    new = splat(color, depth, cam)
    old_img, old_hole = splat_legacy(color, depth, F_PX, torch.eye(3), torch.tensor([0.2, 0.0, 0.0]))
    rows = slice(H // 2 - 40, H // 2 + 40)  # rows well inside the box

    def box_width(res_img, res_hole):
        # the box is the only reddish thing: r clearly above g and b
        red = (res_img[0] > 1.6 * res_img[1]) & (res_img[0] > 1.6 * res_img[2]) & ~res_hole
        return red[rows].float().sum(1).mean().item()

    wn, wo = box_width(new.image, new.hole), box_width(old_img, old_hole)
    print(f"\nbox width v2 {wn:.2f} px, legacy {wo:.2f} px")
    assert abs(wn - wo) <= 1.0  # two edges, 0.5 px each


@pytest.mark.parametrize("yaw", [3.0, 8.0])
def test_psnr_on_visible_pixels(yaw):
    scene = scene_room()
    color, depth = scene_source(scene)
    R, t = orbit_pose(yaw, 0, 2.5)
    cam = make_camera(R, t, F_PX, W, H)
    gt, _, _ = render(scene, cam)
    vis = visible_in_both(scene, SRC, cam)
    new = splat(color, depth, cam)
    old_img, old_hole = splat_legacy(color, depth, F_PX, R, t)
    mask = vis & ~new.hole.cpu() & ~old_hole.cpu()
    p_new, p_old = psnr(new.image.cpu(), gt, mask), psnr(old_img.cpu(), gt, mask)
    print(f"\nyaw {yaw}: PSNR on visible pixels legacy {p_old:.1f} dB, v2 {p_new:.1f} dB")
    assert p_new >= 30.0


def test_disocclusion_strip_stays_open():
    scene = scene_two_layers()
    color, depth = scene_source(scene)
    cam = cam_shift(0.2)
    new = splat(color, depth, cam)
    vis = visible_in_both(scene, SRC, cam)
    _, _, hit = render(scene, cam)
    row = H // 2
    right_edge = int((hit[row] == 1).nonzero().max())
    strip_w = F_PX * 0.2 * (1 / 1.5 - 1 / 4)
    cols = slice(right_edge + 3, right_edge + int(strip_w) - 3)
    assert not vis[row, cols].any()  # sanity: these are true disocclusions
    rows = slice(H // 2 - 40, H // 2 + 40)
    strip = new.hole.cpu()[rows, cols]
    assert strip.float().mean() > 0.97  # no smear bridging foreground and wall


def test_stretched_region_brightness_unchanged():
    g = torch.Generator().manual_seed(0)
    color = torch.rand(3, H, W, generator=g).to(DEV)
    ys = torch.arange(H, dtype=torch.float32, device=DEV)[:, None].expand(H, W)
    depth = 2.0 + (ys - H / 2).clamp(min=0) * 0.012  # tilted ground plane receding downwards... then up
    R, t = orbit_pose(12, 0, 2.5)
    cam = make_camera(R, t, F_PX, W, H)
    res = splat(color, depth, cam, options=SplatOptions(extra_channels=True))
    stretched = (res.extra[1] > 0.5) | (res.extra[0] < 0)  # flagged stretch (may be empty at this angle)
    ok = ~res.hole
    assert abs(float(res.image[:, ok].mean()) - float(color.mean())) / float(color.mean()) < 0.01
    # and the same for a strongly stretched plane (grazing floor)
    floor_depth = 1.0 + 40.0 / (ys.clamp(min=H / 2 + 8) - H / 2)  # distance grows towards the horizon
    res2 = splat(color, floor_depth.to(DEV), cam, options=SplatOptions(extra_channels=True))
    ok2 = ~res2.hole
    assert abs(float(res2.image[:, ok2].mean()) - float(color.mean())) / float(color.mean()) < 0.01
    del stretched


def test_extra_flags_mark_discontinuity():
    color, depth = build_plane_square()
    res = splat(color, depth, cam_shift(0.1), options=SplatOptions(extra_channels=True))
    assert res.extra.shape == (2, H, W)
    # the square (cols 200..299 at 1 m) moves 50 px left; its silhouette carries the discontinuity flag
    assert float(res.extra[0, 200, 146:156].max()) > 0.5
    assert float(res.extra[0, 200, 190:200].max()) < 0.05  # interior of the square: no flag
    assert float(res.extra[1].max()) < 0.05  # nothing is over-stretched in this scene


def test_thin_structures_survive():
    # a 1-px-wide foreground pole must still be visible after a small shift
    color = torch.full((3, H, W), 0.5, device=DEV)
    depth = torch.full((H, W), 4.0, device=DEV)
    color[:, :, 250] = torch.tensor([1.0, 0, 0], device=DEV)[:, None]
    depth[:, 250] = 1.5
    res = splat(color, depth, cam_shift(0.02))
    red = (res.image[0] > 0.8) & (res.image[1] < 0.2) & ~res.hole
    assert red[H // 2].sum() >= 1


@pytest.mark.gpu
def test_speed_12mp():
    W12, H12, f = 4000, 3000, 3000.0
    g = torch.Generator(device="cuda").manual_seed(0)
    color = torch.rand(3, H12, W12, device="cuda", generator=g)
    ys = torch.arange(H12, device="cuda", dtype=torch.float32)[:, None].expand(H12, W12)
    xs = torch.arange(W12, device="cuda", dtype=torch.float32)[None, :].expand(H12, W12)
    depth = 2.0 + 6.0 * (ys / H12) + 0.5 * torch.sin(xs / 300.0)  # smooth, like a real depth map
    depth[1000:2000, 1500:2500] = 1.5  # a foreground object
    R, t = orbit_pose(8, 0, 3.0)
    cam = make_camera(R, t, f, W12, H12)
    splat(color, depth, cam)  # warm-up
    times, stages = [], {}
    for _ in range(7):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        splat(color, depth, cam, timings=stages)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    per_stage = {k: round(v / 7) for k, v in stages.items()}
    print(f"\n12 MP splat v2: median {med:.0f} ms; per-stage ms {per_stage}")
    assert med <= 150
