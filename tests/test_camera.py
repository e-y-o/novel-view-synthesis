import json
from pathlib import Path

import pytest
import torch

from nvs.core.camera import (
    choose_pivot,
    make_camera,
    orbit_pose,
    project,
    transform_to_target,
    unproject,
)

GOLDEN = Path(__file__).parent / "golden" / "camera_vectors.json"


def _project_point(P, R, t, f, W, H):
    Q = transform_to_target(torch.as_tensor(P, dtype=torch.float32), R, t)
    return project(Q[0], Q[1], Q[2], f, W, H)


def test_roundtrip_property():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        W, H = int(torch.randint(8, 2000, (1,), generator=g)), int(torch.randint(8, 2000, (1,), generator=g))
        f = float(torch.rand(1, generator=g) * 3000 + 100)
        u = torch.rand(100, generator=g, dtype=torch.float64) * (W - 1)
        v = torch.rand(100, generator=g, dtype=torch.float64) * (H - 1)
        z = torch.rand(100, generator=g, dtype=torch.float64) * 50 + 0.1
        u2, v2 = project(*unproject(u, v, z, f, W, H), f, W, H)
        assert torch.allclose(u, u2, atol=1e-9) and torch.allclose(v, v2, atol=1e-9)


def test_optical_axis_hits_principal_point():
    u, v = project(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(7.0), 500.0, 640, 480)
    assert (float(u), float(v)) == (319.5, 239.5)
    X, Y, Z = unproject(torch.tensor(99.5), torch.tensor(74.5), torch.tensor(3.0), 120.0, 200, 150)
    assert (float(X), float(Y), float(Z)) == (0.0, 0.0, 3.0)


def test_identity_orbit():
    R, t = orbit_pose(0, 0, 4.0)
    assert torch.allclose(R, torch.eye(3), atol=1e-6) and torch.allclose(t, torch.zeros(3), atol=1e-6)


def test_yaw_sign_and_parallax():
    R, t = orbit_pose(10, 0, 5.0)
    assert t[0] > 0  # positive yaw: camera moves right
    un, _ = _project_point([0, 0, 2.0], R, t, 500.0, 640, 480)
    uf, _ = _project_point([0, 0, 10.0], R, t, 500.0, 640, 480)
    assert un < uf  # near content shifts left relative to far content


def test_pitch_sign_and_parallax():
    R, t = orbit_pose(0, 10, 5.0)
    assert t[1] < 0  # positive pitch: camera moves up (y is down)
    _, vn = _project_point([0, 0, 2.0], R, t, 500.0, 640, 480)
    _, vf = _project_point([0, 0, 10.0], R, t, 500.0, 640, 480)
    assert vn > vf  # near content shifts down relative to far content


@pytest.mark.parametrize("yaw,pitch,dolly", [(0, 0, 0), (12, 0, 0), (-20, 8, 0), (15, -10, 1.5), (5, 5, -2)])
def test_pivot_stays_at_principal_point(yaw, pitch, dolly):
    W, H, f = 640, 480, 500.0
    R, t = orbit_pose(yaw, pitch, 4.0, dolly=dolly)
    u, v = _project_point([0, 0, 4.0], R, t, f, W, H)
    assert abs(float(u) - (W - 1) / 2) < 1e-3 and abs(float(v) - (H - 1) / 2) < 1e-3
    assert abs(float(torch.linalg.det(R.double())) - 1) < 1e-6
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-6)


def test_dolly_moves_closer_and_clamps():
    _, t = orbit_pose(0, 0, 4.0, dolly=1.0)
    assert t[2] == pytest.approx(1.0)
    _, t = orbit_pose(0, 0, 4.0, dolly=100.0)
    assert t[2] == pytest.approx(0.95 * 4.0)
    _, t = orbit_pose(0, 0, 4.0, dolly=-2.0)
    assert t[2] == pytest.approx(-2.0)


def test_pan_is_lateral_in_camera_axes():
    R, t = orbit_pose(0, 0, 4.0, pan=(0.5, -0.25))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)
    assert torch.allclose(t, torch.tensor([0.5, -0.25, 0.0]), atol=1e-6)


def test_orbit_matches_m0_up_to_yaw_sign():
    from nvs.core.splat import orbit_pose as m0

    R0, t0 = m0(-12.0, 4.0)  # M0 positive yaw moved the camera left; M1 positive yaw moves it right
    R1, t1 = orbit_pose(12.0, 0, 4.0)
    assert torch.allclose(R0, R1, atol=1e-6) and torch.allclose(t0, t1, atol=1e-5)


def test_choose_pivot():
    depth = torch.cat([torch.full((80,), 3.0), torch.arange(1.0, 21.0)]).reshape(10, 10)
    assert choose_pivot(depth) == pytest.approx(torch.quantile(depth.flatten().double(), 0.2).item())
    sky = torch.zeros(10, 10, dtype=torch.bool)
    sky[:8] = True  # the first 80 pixels (the 3.0 block) are sky
    expected = torch.quantile(torch.arange(1.0, 21.0).double(), 0.2).item()
    assert choose_pivot(depth, sky) == pytest.approx(expected)
    assert choose_pivot(depth * 100, cap_m=5.0) == 5.0  # capped
    assert choose_pivot(torch.full((4, 4), 3.0), torch.ones(4, 4, dtype=torch.bool)) == 5.0  # all sky
    assert choose_pivot(torch.full((4, 4), float("inf"))) == 5.0  # nothing finite


def test_make_camera_validation():
    R, t = orbit_pose(10, 5, 3.0)
    cam = make_camera(R, t, 800.0, 640, 480)
    assert cam.width == 640 and cam.R.dtype == torch.float32
    with pytest.raises(ValueError):
        make_camera(torch.diag(torch.tensor([1.0, 1.0, -1.0])), t, 800.0, 640, 480)  # reflection
    with pytest.raises(ValueError):
        make_camera(R * 1.1, t, 800.0, 640, 480)  # not orthonormal
    with pytest.raises(ValueError):
        make_camera(R, t, 0.0, 640, 480)
    with pytest.raises(ValueError):
        make_camera(R, t, 800.0, 0, 480)
    with pytest.raises(ValueError):
        make_camera(torch.eye(2), t, 800.0, 640, 480)


def test_golden_vectors():
    data = json.loads(GOLDEN.read_text())
    assert len(data["cases"]) >= 30
    d = torch.float64
    for case in data["cases"]:
        W, H, f = case["width"], case["height"], case["f_px"]
        R, t = torch.tensor(case["R"], dtype=d), torch.tensor(case["t"], dtype=d)
        if "orbit" in case:
            o = case["orbit"]
            R2, t2 = orbit_pose(o["yaw_deg"], o["pitch_deg"], o["pivot_z"], o["dolly"], tuple(o["pan"]))
            assert torch.allclose(R2.double(), R, atol=1e-6) and torch.allclose(t2.double(), t, atol=1e-6)
        px = case["pixels"]
        u, v, z = (torch.tensor([p[k] for p in px], dtype=d) for k in ("u", "v", "z"))
        P = torch.stack(unproject(u, v, z, f, W, H), dim=-1)
        Q = transform_to_target(P, R, t)
        ut, vt = project(Q[:, 0], Q[:, 1], Q[:, 2], f, W, H)
        for got, key in ((ut, "u_t"), (vt, "v_t"), (Q[:, 2], "z_t")):
            want = torch.tensor([p[key] for p in px], dtype=d)
            assert (got - want).abs().max() < 1e-4, (case["id"], key)
