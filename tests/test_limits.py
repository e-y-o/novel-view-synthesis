import statistics
import time

import pytest
import torch
from synth import render, scene_room, scene_sky_tree, scene_two_layers

from nvs.core.camera import make_camera, orbit_pose
from nvs.render.holes import classify_holes
from nvs.render.limits import HARD_WARN_FRACTION, estimate_coverage, limit_orbit
from nvs.render.splat import SplatOptions, splat

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, H, F_PX = 1024, 768, 1000.0
SRC = make_camera(torch.eye(3), torch.zeros(3), F_PX, W, H)


def source(scene):
    color, depth, _ = render(scene, SRC)
    depth = torch.where(torch.isinf(depth), torch.full_like(depth, 40.0), depth)  # sky clamp
    return color.to(DEV), depth.to(DEV)


def full_res_coverage(color, depth, R, t):
    cam = make_camera(R, t, F_PX, W, H)
    res = splat(color, depth, cam, options=SplatOptions(extra_channels=True))
    return classify_holes(res, depth, cam).coverage


def test_identity_has_zero_coverage():
    _, depth = source(scene_two_layers())
    cov = estimate_coverage(depth, F_PX, torch.eye(3), torch.zeros(3))
    assert cov["total"] == 0 and set(cov) == {"disocclusion", "stretch", "out_of_frame", "total"}


def test_coverage_grows_with_yaw():
    _, depth = source(scene_two_layers())
    totals = []
    for yaw in (0, 2, 4, 8, 12):
        R, t = orbit_pose(yaw, 0, 2.5)
        totals.append(estimate_coverage(depth, F_PX, R, t)["total"])
    print("\ncoverage vs yaw:", [round(x, 3) for x in totals])
    assert all(b > a for a, b in zip(totals, totals[1:], strict=False))


@pytest.mark.parametrize(
    "name,scene,pivot", [("room", scene_room(), 2.5), ("sky_tree", scene_sky_tree(), 3.5)]
)
def test_reduced_estimate_matches_full_resolution(name, scene, pivot):
    color, depth = source(scene)
    for yaw in (3.0, 8.0, 12.0):
        R, t = orbit_pose(yaw, 0, pivot)
        small = estimate_coverage(depth, F_PX, R, t, max_mp=0.25)["total"]
        full = full_res_coverage(color, depth, R, t)["total"]
        print(f"\n{name} yaw {yaw}: 0.25 MP {small:.3f} vs full {full:.3f}")
        assert abs(small - full) <= 0.03


def test_limit_orbit_small_move_is_unlimited():
    _, depth = source(scene_two_layers())
    r = limit_orbit(depth, F_PX, 1.0, 0.0, 2.5)
    assert (r.limited, r.scale) == (False, 1.0)
    assert r.yaw_deg == 1.0 and r.hole_fraction <= 0.12


def test_limit_orbit_large_move_is_limited():
    _, depth = source(scene_room())
    r = limit_orbit(depth, F_PX, 30.0, 10.0, 2.5, dolly=0.5, pan=(0.2, 0.0))
    print(f"\nlimited to scale {r.scale:.3f}: yaw {r.yaw_deg:.1f}, fraction {r.hole_fraction:.3f}")
    assert r.limited and 0 < r.scale < 1
    assert r.yaw_deg == pytest.approx(30.0 * r.scale) and r.pitch_deg == pytest.approx(10.0 * r.scale)
    assert r.hole_fraction <= 0.12 + 0.01
    # the reported fraction is what the full estimate says at that pose
    R, t = orbit_pose(r.yaw_deg, r.pitch_deg, 2.5, r.dolly, r.pan)
    assert estimate_coverage(depth, F_PX, R, t)["total"] == pytest.approx(r.hole_fraction, abs=1e-6)
    loose = limit_orbit(depth, F_PX, 30.0, 10.0, 2.5, max_hole_fraction=HARD_WARN_FRACTION)
    assert loose.scale >= r.scale


def test_count_out_of_frame_option():
    """Excluding out-of-frame area lets a move go further; the reported fraction is the counted one."""
    _, depth = source(scene_two_layers())
    args = (depth, F_PX, 25.0, 0.0, 2.5)
    counted = limit_orbit(*args)
    excluded = limit_orbit(*args, count_out_of_frame=False)
    assert counted.limited and excluded.scale >= counted.scale
    assert limit_orbit(*args, count_out_of_frame=True).scale == counted.scale  # the default is True
    R, t = orbit_pose(excluded.yaw_deg, excluded.pitch_deg, 2.5, excluded.dolly, excluded.pan)
    cov = estimate_coverage(depth, F_PX, R, t)
    assert cov["total"] - cov["out_of_frame"] == pytest.approx(excluded.hole_fraction, abs=1e-6)
    assert excluded.hole_fraction <= 0.12 + 0.01


def test_hole_fraction_monotone_in_scale():
    _, depth = source(scene_room())
    prev = -1.0
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        R, t = orbit_pose(20 * s, 5 * s, 2.5)
        frac = estimate_coverage(depth, F_PX, R, t)["total"]
        assert frac >= prev - 1e-6
        prev = frac


@pytest.mark.gpu
def test_estimate_speed():
    _, depth = source(scene_room())
    R, t = orbit_pose(8, 0, 2.5)
    estimate_coverage(depth, F_PX, R, t)  # warm-up
    times = []
    for _ in range(15):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        estimate_coverage(depth, F_PX, R, t)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    print(f"\nestimate_coverage (0.25 MP, GPU): median {med:.1f} ms")
    assert med <= 15
