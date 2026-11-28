import time

import pytest
import torch
from synth import psnr, render, scene_room, scene_sky_tree, scene_two_layers, visible_in_both

from nvs.core.camera import make_camera, orbit_pose

W, H, F = 512, 384, 500.0
SRC = make_camera(torch.eye(3), torch.zeros(3), F, W, H)


def test_identity_render_is_deterministic():
    for scene in (scene_two_layers(), scene_room(), scene_sky_tree()):
        a, b = render(scene, SRC), render(scene, SRC)
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)
    assert (
        scene_two_layers().primitives[0].texture.shift.equal(scene_two_layers().primitives[0].texture.shift)
    )


def test_box_centre_depth_is_analytic():
    _, depth, hit = render(scene_two_layers(), SRC)
    cy, cx = H // 2, W // 2  # principal point is at (W-1)/2: use the pixel next to it
    assert depth[cy, cx] == pytest.approx(1.5, abs=1e-3)
    assert hit[cy, cx] == 1


def test_plane_depth_is_z_depth_not_ray_length():
    _, depth, hit = render(scene_two_layers(), SRC)
    # corners miss the box (1 m wide at 1.5 m) and hit the wall: z-depth stays 4 m
    for y, x in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        assert hit[y, x] == 0
        assert depth[y, x] == pytest.approx(4.0, abs=1e-4)


def test_sky_and_values():
    color, depth, hit = render(scene_sky_tree(), SRC)
    sky = hit == -1
    assert sky.any() and torch.isinf(depth[sky]).all()
    assert color.min() >= 0 and color.max() <= 1


def test_visible_in_both_identity():
    for scene in (scene_two_layers(), scene_room(), scene_sky_tree()):
        assert visible_in_both(scene, SRC, SRC).all()


def test_visible_in_both_strip_next_to_box():
    shift = 0.2
    tgt = make_camera(torch.eye(3), torch.tensor([shift, 0.0, 0.0]), F, W, H)
    scene = scene_two_layers()
    vis = visible_in_both(scene, SRC, tgt)
    _, _, hit = render(scene, tgt)
    row = H // 2
    box_cols = torch.nonzero(hit[row] == 1).flatten()
    right_edge = int(box_cols.max())
    run = 0  # contiguous not-visible wall pixels right of the box
    x = right_edge + 1
    while x < W and not vis[row, x]:
        run += 1
        x += 1
    expected = F * shift * (1 / 1.5 - 1 / 4)
    assert abs(run - expected) <= 1.0, (run, expected)
    # box pixels themselves are visible from the source
    assert vis[row, box_cols].all()
    # the right image border shows wall beyond the source frame
    assert not vis[row, W - 1]


def test_splat_baseline_psnr_helper():
    a = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(0))
    assert psnr(a, a) == float("inf")
    assert psnr(a, a + 0.1) == pytest.approx(20.0, abs=1e-4)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:4] = True
    b = a.clone()
    b[:, 4:] += 0.5
    assert psnr(a, b, mask) == float("inf")


def test_orbit_target_has_hidden_pixels_and_is_fast():
    R, t = orbit_pose(8, 0, 2.5)
    tgt = make_camera(R, t, F, W, H)
    scene = scene_room()
    t0 = time.perf_counter()
    render(scene, tgt)
    assert time.perf_counter() - t0 < 2.0
    vis = visible_in_both(scene, SRC, tgt)
    assert 0.5 < vis.float().mean() < 1.0
