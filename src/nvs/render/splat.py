"""Forward splatting (single-image novel view synthesis) in pure PyTorch.

``splat`` warps a colour image to a target camera using its depth map. Conventions (OpenCV; see
``nvs.core.camera``): x right, y down, z forward, source camera = identity, target pose
``P_target = R^T (P - t)``. The target camera has the same intrinsics and size as the source.

How it works, in plain language
-------------------------------
Every source pixel is lifted to 3D, moved into the target camera and dropped on the target grid. Two passes:
the first finds, for each target pixel, the nearest surface that lands on it (``zbuf``); the second adds up
colours with weights that fade out quickly for anything behind that nearest surface (``sigma``), so front
objects hide what is behind them. Dividing by the accumulated weight gives the colour; where almost nothing
landed (weight < ``eps``) the pixel is a hole.

*Footprints.* When the camera moves, a surface seen at a grazing angle is stretched: neighbouring source
pixels land several target pixels apart and leave pinholes. We measure that spread (distance in the target
between a pixel and its right/bottom neighbour) and let each source pixel cover a bigger patch:

* radius <= 1: plain bilinear 2x2 splat (the M0 kernel);
* 1 < radius <= 2: 3x3 tent kernel centred on the nearest target pixel;
* 2 < radius <= ``r_max``: 5x5 tent kernel.

The tent has peak weight 1, so a stretched region ends up with total weight ~1 per target pixel (each
source pixel contributes about 1/r^2 per pixel it covers, and there are r^2 target pixels per source pixel)
and the normalised colour does not change brightness. Beyond ``r_max`` we do not paint further: what is left
is a hole that ``holes.classify_holes`` labels "stretch".

*Discontinuities.* If a pixel and its neighbour differ in depth by more than ``gap_ratio`` they belong to
different surfaces (object edge). Across such a pair we do not stretch: the radius falls back to 1, so the gap
stays open instead of being bridged by a smear between foreground and background.

Geometry and accumulation are float32 (fp16 atomics are imprecise and not faster here).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from nvs.core.types import CameraSpec, SplatResult

Z_CLAIM_MIN_WEIGHT = 0.1  # a contribution lighter than this does not claim the z-buffer for a target pixel


@dataclass
class SplatOptions:
    sigma: float = 0.03  # depth weighting: weight ~ exp(-(z - z_front) / (sigma * z_front))
    eps: float = 1e-2  # hole threshold on accumulated weight
    r_max: float = 3.0  # max footprint radius in target pixels (kernels are built up to 5x5)
    gap_ratio: float = 1.08  # neighbour depth ratio treated as a discontinuity
    extra_channels: bool = False  # also return warped [discontinuity, stretch] flags first in ``extra``


def _project(depth: torch.Tensor, R: torch.Tensor, t: torch.Tensor, f_px: float):
    """Target-image position (up, vp) [px] and target z (Zp) [m] of every source pixel, as [H, W] maps."""
    dev = depth.device
    H, W = depth.shape
    cx, cy = (W - 1) / 2, (H - 1) / 2
    v, u = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    dx = (u - cx) / f_px * depth - t[0]
    dy = (v - cy) / f_px * depth - t[1]
    dz = depth - t[2]
    Xp = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz  # R^T (P - t)
    Yp = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    Zp = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
    up = f_px * Xp / Zp + cx
    vp = f_px * Yp / Zp + cy
    valid = (Zp > 1e-3) & torch.isfinite(up) & torch.isfinite(vp)
    return up, vp, Zp, valid


def _pair(up, vp, valid, depth, axis: int, gap_ratio: float):
    """Spread (target px) and discontinuity flag between each pixel and its next neighbour along ``axis``.

    Returns (spread[H, W], disc[H, W] bool); the last row/column has no neighbour and gets spread 1 / no gap.
    """
    sl_a = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
    sl_b = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
    d_a, d_b = depth[sl_a], depth[sl_b]
    ok = valid[sl_a] & valid[sl_b]
    spread = torch.hypot(up[sl_b] - up[sl_a], vp[sl_b] - vp[sl_a])
    ratio = torch.maximum(d_a, d_b) / torch.minimum(d_a, d_b).clamp_min(1e-6)
    disc = ratio > gap_ratio
    spread = torch.where(ok & ~disc, spread, torch.ones_like(spread))
    pad = (0, 1, 0, 0) if axis == 1 else (0, 0, 0, 1)
    spread = torch.nn.functional.pad(spread, pad, value=1.0)
    disc = torch.nn.functional.pad(disc, pad, value=False)
    return spread, disc


def _footprint(up, vp, valid, depth, opts: SplatOptions):
    """Per-source-pixel footprint radius r in [1, r_max] plus the discontinuity and over-stretch flags."""
    sx, dx = _pair(up, vp, valid, depth, 1, opts.gap_ratio)
    sy, dy = _pair(up, vp, valid, depth, 0, opts.gap_ratio)
    spread = torch.maximum(sx, sy)
    stretch = spread > opts.r_max
    r = spread.clamp(1.0, opts.r_max)
    # a pixel also borders the discontinuity from the other side (its left / upper neighbour)
    disc = dx | dy
    disc[:, 1:] |= dx[:, :-1]
    disc[1:, :] |= dy[:-1, :]
    return r, disc, stretch


TERM_CHUNK = 8_000_000  # max (source pixel x kernel tap) pairs processed per batch


def _batched_terms(kind, ub, vb, rb, H, W):
    """Yield (slice, idx, w, ok, taps) for chunks of source pixels.

    All kernel taps of a chunk are laid out as one flat batch ``[taps * m]`` (tap-major), so a bucket costs a
    handful of kernel launches per chunk instead of dozens per tap. ``kind`` is "bil" (2x2 bilinear), 3 or 5
    (tent kernels of that width, radius ``rb``, peak weight 1, centred on the nearest target pixel).
    """
    if kind == "bil":
        taps = 4
        ox = torch.tensor([0, 1, 0, 1], device=ub.device)
        oy = torch.tensor([0, 0, 1, 1], device=ub.device)
    else:
        k = 1 if kind == 3 else 2
        rng = torch.arange(-k, k + 1, device=ub.device)
        oy, ox = torch.meshgrid(rng, rng, indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)
        taps = ox.numel()
    n = ub.shape[0]
    step = max(1, TERM_CHUNK // taps)
    for s in range(0, n, step):
        sl = slice(s, min(n, s + step))
        u, v = ub[sl][None], vb[sl][None]
        if kind == "bil":
            x0, y0 = torch.floor(u), torch.floor(v)
            fx, fy = u - x0, v - y0
            xi, yi = x0 + ox[:, None], y0 + oy[:, None]
            w = torch.where(ox[:, None] == 1, fx, 1 - fx) * torch.where(oy[:, None] == 1, fy, 1 - fy)
        else:
            r = rb[sl][None]
            xi, yi = torch.round(u) + ox[:, None], torch.round(v) + oy[:, None]
            w = (1 - (xi - u).abs() / r).clamp_(min=0) * (1 - (yi - v).abs() / r).clamp_(min=0)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (w > 0)
        idx = yi.clamp(0, H - 1).long() * W + xi.clamp(0, W - 1).long()
        yield sl, idx.reshape(-1), w.reshape(-1), ok.reshape(-1), taps


def splat(
    color: torch.Tensor,
    depth: torch.Tensor,
    cam: CameraSpec,
    *,
    extra: torch.Tensor | None = None,
    options: SplatOptions | None = None,
    src_f_px: float | None = None,
    timings: dict | None = None,
) -> SplatResult:
    """Warp ``color`` [C, H, W] with ``depth`` [H, W] (z-depth, metres, finite) to the target camera ``cam``.

    ``cam`` must have the source's size; its ``f_px`` is also the source focal length unless ``src_f_px`` is
    given. ``extra`` [K, H, W] is accumulated exactly like colour and returned normalised. With
    ``options.extra_channels`` the result's ``extra`` starts with two warped flag channels
    [discontinuity, stretch] (values in [0, 1]) followed by the caller's ``extra``.
    """
    opts = options or SplatOptions()
    dev = color.device
    C, H, W = color.shape
    if depth.shape != (H, W):
        raise ValueError(f"depth {tuple(depth.shape)} does not match colour {(H, W)}")
    if (cam.width, cam.height) != (W, H):
        raise ValueError(f"camera is {cam.width}x{cam.height} but the source is {W}x{H}")
    R = cam.R.to(device=dev, dtype=torch.float32)
    t = cam.t.to(device=dev, dtype=torch.float32)
    f_src = float(src_f_px if src_f_px is not None else cam.f_px)
    depth = depth.to(torch.float32)

    def tick(name, t0):
        if timings is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    up, vp, Zp, valid = _project(depth, R, t, f_src)
    r, disc, stretch = _footprint(up, vp, valid, depth, opts)
    keep = valid & (up > -1) & (up < W) & (vp > -1) & (vp < H)
    chans = [color.reshape(C, -1).to(torch.float32)]
    if opts.extra_channels:
        chans.append(torch.stack([disc.reshape(-1), stretch.reshape(-1)]).to(torch.float32))
    if extra is not None:
        chans.append(extra.reshape(extra.shape[0], -1).to(device=dev, dtype=torch.float32))
    keep_f = keep.reshape(-1)
    payload = torch.cat(chans, 0)[:, keep_f].T.contiguous()  # [n, P]: rows are contiguous for index_add_
    P = payload.shape[1]
    up_k, vp_k, Z_k, r_k = (a.reshape(-1)[keep_f] for a in (up, vp, Zp, r))
    del up, vp, Zp, r, disc, stretch, keep
    tick("project", t0)

    # bucket by footprint radius: 2x2 bilinear, 3x3 tent, 5x5 tent
    buckets = []
    for kind, sel in (("bil", r_k <= 1.0), (3, (r_k > 1.0) & (r_k <= 2.0)), (5, r_k > 2.0)):
        idx = sel.nonzero().squeeze(1)
        if idx.numel():
            buckets.append((kind, up_k[idx], vp_k[idx], Z_k[idx], r_k[idx], payload[idx]))
    del payload

    t0 = time.perf_counter()
    zmin = torch.full((H * W,), float("inf"), device=dev)
    for kind, ub, vb, Zb, rb, _ in buckets:
        for sl, idx, w, ok, taps in _batched_terms(kind, ub, vb, rb, H, W):
            Zf = Zb[sl].repeat(taps)
            claim = ok & (w >= Z_CLAIM_MIN_WEIGHT)
            zmin.scatter_reduce_(0, idx, torch.where(claim, Zf, torch.full_like(Zf, float("inf"))), "amin")
    tick("pass1_zmin", t0)

    t0 = time.perf_counter()
    acc = torch.zeros(H * W, P + 1, device=dev)
    for kind, ub, vb, Zb, rb, pay in buckets:
        for sl, idx, w, ok, taps in _batched_terms(kind, ub, vb, rb, H, W):
            Zf = Zb[sl].repeat(taps)
            zm = zmin[idx]
            # surfaces behind the front one fade out; lighter-than-claim contributions in front keep weight w
            wt = torch.where(
                ok, w * torch.exp(-(Zf - zm).clamp_(min=0) / (opts.sigma * zm)), torch.zeros_like(w)
            )
            src = torch.cat([pay[sl].repeat(taps, 1) * wt[:, None], wt[:, None]], 1)
            acc.index_add_(0, idx, src)
    tick("pass2_accum", t0)

    t0 = time.perf_counter()
    wsum = acc[:, P]
    hole = wsum < opts.eps
    norm = (acc[:, :P] / wsum.clamp_min(1e-8)[:, None]).T.contiguous()  # [P, H*W]
    norm[:, hole] = 0
    s = ((wsum - opts.eps) / (3 * opts.eps)).clamp_(0, 1)
    alpha = s * s * (3 - 2 * s)
    zbuf = torch.where(hole, torch.full_like(zmin, float("inf")), zmin)
    out_extra = norm[C:].reshape(P - C, H, W) if P > C else None
    tick("normalise", t0)
    return SplatResult(
        image=norm[:C].reshape(C, H, W),
        alpha=alpha.reshape(H, W),
        hole=hole.reshape(H, W),
        zbuf=zbuf.reshape(H, W),
        extra=out_extra,
    )


# --------------------------------------------------------------------------------------------------------
# M0 prototype, kept for comparison tests. Deprecated: removed in T14.
# --------------------------------------------------------------------------------------------------------


def splat_legacy(color, depth, f_px, R, t, sigma=0.03, eps=1e-2, timings=None):
    """M0 splatter (bilinear 2x2 only), kept for comparison tests. Deprecated: removed in T14."""
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
    v, u = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
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
        zmin.scatter_reduce_(
            0, idx, torch.where(ok, Zp, torch.full_like(Zp, float("inf"))), "amin", include_self=True
        )
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


def orbit_pose_legacy(yaw_deg, pivot_z):
    """M0 orbit: positive yaw moves the camera LEFT (nvs.core.camera.orbit_pose moves it right).
    Camera orbits about the point (0,0,pivot_z) by yaw and keeps looking at it. Pure rotation about the
    camera centre has no parallax (no holes), so the benchmark/renders use an orbit."""
    a = math.radians(yaw_deg)
    R = torch.tensor(
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=torch.float32
    )
    p = torch.tensor([0.0, 0.0, pivot_z])
    return R, p - R @ p
