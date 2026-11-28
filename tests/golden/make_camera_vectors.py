"""Generate tests/golden/camera_vectors.json (run manually with `uv run python` on this file).

Expected values use an independent pure-Python float64 implementation of the M1 camera convention so the
TypeScript renderer (M2) can be checked against the same file. Cases are seeded and deterministic.
"""

import json
import math
import random
from pathlib import Path

from nvs.core.camera import orbit_pose

SEED = 20260101
N_ORBIT, N_RANDOM, N_PIXELS = 24, 12, 8


def rot_axis_angle(axis, angle):
    n = math.sqrt(sum(a * a for a in axis))
    x, y, z = (a / n for a in axis)
    c, s = math.cos(angle), math.sin(angle)
    k = 1 - c
    return [
        [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
        [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
        [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
    ]


def expected(case, rng):
    W, H, f = case["width"], case["height"], case["f_px"]
    R, t = case["R"], case["t"]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    pixels = []
    while len(pixels) < N_PIXELS:
        u, v, z = rng.uniform(0, W - 1), rng.uniform(0, H - 1), rng.uniform(0.5, 30.0)
        P = [(u - cx) / f * z, (v - cy) / f * z, z]
        d = [P[i] - t[i] for i in range(3)]
        Q = [sum(R[r][c] * d[r] for r in range(3)) for c in range(3)]  # R^T d
        if Q[2] < 0.1:
            continue
        pixels.append(
            {"u": u, "v": v, "z": z, "u_t": f * Q[0] / Q[2] + cx, "v_t": f * Q[1] / Q[2] + cy, "z_t": Q[2]}
        )
    return pixels


def main():
    rng = random.Random(SEED)
    cases = []
    for i in range(N_ORBIT + N_RANDOM):
        W, H = rng.choice([(640, 480), (1000, 750), (480, 640), (1920, 1080), (301, 203)])
        case = {"id": i, "width": W, "height": H, "f_px": rng.uniform(0.6, 2.0) * max(W, H)}
        if i < N_ORBIT:
            orbit = {
                "yaw_deg": rng.uniform(-30, 30),
                "pitch_deg": rng.uniform(-20, 20),
                "pivot_z": rng.uniform(1.0, 8.0),
                "dolly": rng.uniform(-1.0, 3.0),
                "pan": [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)],
            }
            R, t = orbit_pose(**{**orbit, "pan": tuple(orbit["pan"])})
            case["orbit"] = orbit
            case["R"] = R.double().tolist()
            case["t"] = t.double().tolist()
        else:
            axis = [rng.gauss(0, 1) for _ in range(3)]
            case["R"] = rot_axis_angle(axis, math.radians(rng.uniform(-40, 40)))
            case["t"] = [rng.uniform(-1, 1) for _ in range(3)]
        case["pixels"] = expected(case, rng)
        cases.append(case)
    out = Path(__file__).with_name("camera_vectors.json")
    doc = {"seed": SEED, "convention": "OpenCV; P_target = R^T (P - t)", "cases": cases}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
