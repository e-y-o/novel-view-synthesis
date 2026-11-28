import statistics
import time

import pytest
import torch
from synth import render, scene_room, scene_two_layers, visible_in_both

from nvs.core.camera import make_camera, orbit_pose
from nvs.core.types import HoleClass, SplatResult
from nvs.render.holes import HoleMap, classify_holes, fill_specks, find_specks
from nvs.render.splat import SplatOptions, splat

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, H, F_PX = 512, 384, 500.0
SRC = make_camera(torch.eye(3), torch.zeros(3), F_PX, W, H)
FLAGS = SplatOptions(extra_channels=True)


def run(scene, cam):
    color, depth, _ = render(scene, SRC)
    color, depth = color.to(DEV), depth.to(DEV)
    res = splat(color, depth, cam, options=FLAGS)
    return res, depth, classify_holes(res, depth, cam)


def shift_cam(dx):
    return make_camera(torch.eye(3), torch.tensor([dx, 0.0, 0.0]), F_PX, W, H)


def small_cam(w, h):
    return make_camera(torch.eye(3), torch.zeros(3), float(w), w, h)


def manual(hole, flags=None):
    h, w = hole.shape
    extra = torch.zeros(2, h, w, device=hole.device) if flags is None else flags
    return SplatResult(
        torch.zeros(3, h, w, device=hole.device), (~hole).float(), hole, torch.ones(h, w), extra
    )


def test_disocclusion_strip_is_class_1():
    scene = scene_two_layers()
    cam = shift_cam(0.2)
    res, depth, hm = run(scene, cam)
    vis = visible_in_both(scene, SRC, cam)
    _, _, hit = render(scene, cam)
    row = H // 2
    right_edge = int((hit[row] == 1).nonzero().max())
    strip_w = int(F_PX * 0.2 * (1 / 1.5 - 1 / 4))
    rows, cols = slice(H // 2 - 40, H // 2 + 40), slice(right_edge + 3, right_edge + strip_w - 3)
    assert not vis[row, cols].any()
    strip = hm.classes.cpu()[rows, cols]
    assert (strip == HoleClass.DISOCCLUSION).float().mean() >= 0.95
    assert not (strip == HoleClass.OUT_OF_FRAME).any()
    assert isinstance(hm, HoleMap) and hm.classes.dtype == torch.uint8 and hm.mask.dtype == torch.float32


def test_yaw_orbit_border_is_class_3_and_box_holes_class_1():
    scene = scene_two_layers()
    R, t = orbit_pose(6, 0, 2.5)
    cam = make_camera(R, t, F_PX, W, H)
    res, depth, hm = run(scene, cam)
    cls = hm.classes.cpu()
    # the wall (beyond the pivot) moves right: the left border strip is outside the source frame
    left = cls[H // 4 : 3 * H // 4, :6]
    assert (left == HoleClass.OUT_OF_FRAME).float().mean() > 0.95
    # holes beside the box (well inside the image) are disocclusions
    _, _, hit = render(scene, cam)
    row = H // 2
    right_edge = int((hit[row] == 1).nonzero().max())
    near_box = cls[H // 2 - 40 : H // 2 + 40, right_edge + 3 : right_edge + 15]
    holes = near_box > 0
    assert holes.any()
    assert (near_box[holes] == HoleClass.DISOCCLUSION).float().mean() > 0.95


def test_stretch_flags_give_class_2_and_disocclusion_wins():
    # Over-stretched holes are rare with the default r_max = 3 (tent footprints cover even very oblique
    # walls), so the flag -> class rule is tested on hand-made flag maps; test_mask_dilation covers the
    # interior of a big hole as well.
    hole = torch.zeros(60, 120, dtype=torch.bool, device=DEV)
    hole[20:40, 10:40] = True  # stretch only
    hole[20:40, 70:100] = True  # stretch flags and a discontinuity flag next to it
    flags = torch.zeros(2, 60, 120, device=DEV)
    flags[1, 15:45, 5:45] = 1.0
    flags[1, 15:45, 65:105] = 1.0
    flags[0, 15:45, 100:105] = 1.0
    flags[:, hole] = 0
    hm = classify_holes(manual(hole, flags), torch.full((60, 120), 3.0, device=DEV), small_cam(120, 60))
    assert (hm.classes[20:40, 10:40] == HoleClass.STRETCH).all()
    # classes are local: next to the discontinuity flag a disocclusion, next to stretch-only flags stretch
    assert (hm.classes[20:40, 96:100] == HoleClass.DISOCCLUSION).all()
    assert (hm.classes[20:40, 70:74] == HoleClass.STRETCH).all()


def test_identity_has_no_holes():
    res, depth, hm = run(scene_room(), SRC)
    assert not res.hole.any()
    assert int(hm.classes.sum()) == 0 and hm.coverage["total"] == 0 and float(hm.mask.max()) == 0


def test_coverage_adds_up():
    R, t = orbit_pose(10, 3, 2.5)
    cam = make_camera(R, t, F_PX, W, H)
    res, depth, hm = run(scene_room(), cam)
    parts = sum(v for k, v in hm.coverage.items() if k != "total")
    assert parts == pytest.approx(hm.coverage["total"], abs=1e-6)
    assert hm.coverage["total"] == pytest.approx(float((hm.classes > 0).float().mean()), abs=1e-6)
    assert set(hm.coverage) == {"disocclusion", "stretch", "out_of_frame", "total"}
    assert hm.coverage["out_of_frame"] > 0


def test_speck_filled_and_removed():
    hole = torch.zeros(40, 50, dtype=torch.bool, device=DEV)
    hole[20, 25] = True
    hole[10, 10:12] = True  # a domino is also a speck
    hole[30:34, 5:9] = True  # a 4x4 block is a real hole
    img = torch.linspace(0, 1, 50, device=DEV)[None, None, :].expand(3, 40, 50).clone()
    hm = classify_holes(manual(hole), torch.full((40, 50), 3.0, device=DEV), small_cam(50, 40))
    speck = find_specks(hole)
    assert speck[20, 25] and speck[10, 10] and speck[10, 11] and not speck[31, 6]
    assert hm.speck is not None and hm.classes[20, 25] == 0 and hm.classes[10, 10] == 0
    assert (hm.classes[30:34, 5:9] > 0).all()
    assert hm.coverage["total"] == pytest.approx(16 / (40 * 50))
    filled = fill_specks(img, hole)
    assert torch.allclose(filled[:, 20, 25], img[:, 20, 25], atol=1 / 49 + 1e-6)  # mean of neighbours
    assert torch.equal(filled[:, 31, 6], img[:, 31, 6])  # real holes untouched
    assert torch.equal(filled[:, 0, 0], img[:, 0, 0])


def test_hole_touching_border_is_not_a_speck():
    hole = torch.zeros(20, 20, dtype=torch.bool, device=DEV)
    hole[0, 5] = True
    hole[10, 10] = True
    sp = find_specks(hole)
    assert not sp[0, 5] and sp[10, 10]


@pytest.mark.parametrize("stretch,grow", [(True, 3), (False, 5)])
def test_mask_dilation(stretch, grow):
    hole = torch.zeros(80, 100, dtype=torch.bool, device=DEV)
    hole[30:50, 40:70] = True
    flags = torch.zeros(2, 80, 100, device=DEV)
    flags[1 if stretch else 0, 25:55, 35:75] = 1.0  # flags next to (not on) the hole
    flags[:, hole] = 0
    hm = classify_holes(manual(hole, flags), torch.full((80, 100), 3.0, device=DEV), small_cam(100, 80))
    want = HoleClass.STRETCH if stretch else HoleClass.DISOCCLUSION
    assert (hm.classes[hole] == want).all()
    m = hm.mask
    assert (m[hole] == 1).all()
    row = 40
    assert m[row, 40 - grow] == 1.0  # the dilated hole reaches `grow` px beyond the original edge
    assert m[row, 40 - grow - 8] < 0.05  # well outside: feathered away
    assert 0.0 < float(m[row, 40 - grow - 1]) < 1.0  # feather ramp just outside the dilated edge
    assert float(m[row, 40 - grow - 3]) < float(m[row, 40 - grow - 1])


@pytest.mark.gpu
def test_speed_12mp():
    W12, H12, f = 4000, 3000, 3000.0
    g = torch.Generator(device="cuda").manual_seed(0)
    color = torch.rand(3, H12, W12, device="cuda", generator=g)
    ys = torch.arange(H12, device="cuda", dtype=torch.float32)[:, None].expand(H12, W12)
    xs = torch.arange(W12, device="cuda", dtype=torch.float32)[None, :].expand(H12, W12)
    depth = 2.0 + 6.0 * (ys / H12) + 0.5 * torch.sin(xs / 300.0)
    depth[1000:2000, 1500:2500] = 1.5
    R, t = orbit_pose(8, 0, 3.0)
    cam = make_camera(R, t, f, W12, H12)
    res = splat(color, depth, cam, options=FLAGS)
    classify_holes(res, depth, cam)  # warm-up
    times = []
    for _ in range(7):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hm = classify_holes(res, depth, cam)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    print(f"\n12 MP classify_holes: median {med:.0f} ms, coverage {hm.coverage}")
    assert med <= 60
