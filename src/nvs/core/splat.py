"""Forward splatting (single-image novel view synthesis) in pure PyTorch. Prototyped in M0 ticket T06.

OpenCV convention (x right, y down, z forward); the source camera is the identity.
splat(color[C,H,W], depth[H,W], f_px, R[3,3], t[3]) -> (image[C,H,W], hole_mask[H,W]); R, t are the target
camera pose in the source frame: P_target = R^T (P - t).
Geometry is float32 and colour accumulation is float32 (fp16 atomics are imprecise and not faster here).
"""
from __future__ import annotations

import math
import time

import torch


def splat(color, depth, f_px, R, t, sigma=0.03, eps=1e-2, timings=None):
    dev = color.device
    C, H, W = color.shape
    R = torch.as_tensor(R, dtype=torch.float32, device=dev)
    t = torch.as_tensor(t, dtype=torch.float32, device=dev)
    cx, cy = (W - 1) / 2, (H - 1) / 2

    def tick(name, t0):
        if timings is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            timings[name] = timings.get(name, 0) + (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                          torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    z = depth.reshape(-1)
    dx = (u.reshape(-1) - cx) / f_px * z - t[0]
    dy = (v.reshape(-1) - cy) / f_px * z - t[1]
    dz = z - t[2]
    Xp = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz  # R^T (P - t)
    Yp = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    Zp = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
    up = f_px * Xp / Zp + cx
    vp = f_px * Yp / Zp + cy
    keep = (Zp > 1e-3) & torch.isfinite(up) & torch.isfinite(vp) & (up > -1) & (up < W) & (vp > -1) & (vp < H)
    up, vp, Zp = up[keep], vp[keep], Zp[keep]
    col = color.reshape(C, -1)[:, keep]
    x0 = torch.floor(up)
    y0 = torch.floor(vp)
    fx = up - x0
    fy = vp - y0
    x0 = x0.long()
    y0 = y0.long()
    corners = []
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi, yi = x0 + ox, y0 + oy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)
        bil = (fx if ox else 1 - fx) * (fy if oy else 1 - fy)
        corners.append((idx, bil, ok))
    tick("project", t0)

    t0 = time.perf_counter()
    zmin = torch.full((H * W,), float("inf"), device=dev)
    for idx, _, ok in corners:
        zmin.scatter_reduce_(0, idx, torch.where(ok, Zp, torch.full_like(Zp, float("inf"))), "amin", include_self=True)
    tick("pass1_zmin", t0)

    t0 = time.perf_counter()
    acc = torch.zeros(C + 1, H * W, device=dev)
    for idx, bil, ok in corners:
        zm = zmin[idx]
        w = torch.where(ok, bil * torch.exp(-(Zp - zm) / (sigma * zm)), torch.zeros_like(bil))
        acc.index_add_(1, idx, torch.cat([col * w, w[None]], 0))
    tick("pass2_accum", t0)

    t0 = time.perf_counter()
    wsum = acc[C]
    hole = wsum < eps
    out = acc[:C] / wsum.clamp_min(1e-8)
    out[:, hole] = 0
    tick("normalise", t0)
    return out.reshape(C, H, W), hole.reshape(H, W)


def orbit_pose(yaw_deg, pivot_z):
    """Camera orbits about the point (0,0,pivot_z) by yaw and keeps looking at it. Pure rotation about the
    camera centre has no parallax (no holes), so the benchmark/renders use an orbit."""
    a = math.radians(yaw_deg)
    R = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=torch.float32)
    p = torch.tensor([0.0, 0.0, pivot_z])
    return R, p - R @ p
