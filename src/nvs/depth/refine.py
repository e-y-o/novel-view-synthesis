"""Depth refinement: sky handling, outlier clamping and RGB-guided upsampling to source resolution.

``refine_depth`` turns a ``RawDepth`` (inference resolution, may contain inf/km values) into a ``DepthResult``
at **source** resolution that is finite everywhere and whose edges follow the photo's edges.

* Sky: ``inf``, ``> 10 km`` or ``> 4 x p99`` of the finite depth. Those pixels become ``4 x p99``. The mask
  is closed with a 3x3 morphological closing (dilate, then erode by 1 px) so isolated pinholes inside the
  sky do not flicker between sky and not-sky.
* Outliers: depth is clamped to ``[0.9 x p0.5, 4 x p99]``: this removes the near-zero spikes Depth Pro 2 can
  produce on tiny regions and leaves normal depth untouched.
* Upsampling: a colour guided filter (He et al., "Fast Guided Filter") on **inverse depth** (perspective is
  linear in 1/z), with the RGB image as guide. Coefficients are solved at inference resolution; the final
  step is evaluated in row bands at full resolution, so peak memory stays bounded for 24 MP sources.
  Where the photo has no edge the filter reduces to a smooth (bilinear-like) upsample: it cannot invent edges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nvs.core.types import DepthResult
from nvs.depth.base import RawDepth

SKY_FACTOR = 4.0
QUANTILE_SAMPLES = 1_000_000
GUIDE_RADIUS = 2  # box radius in inference-resolution pixels (about 2 x the upsampling factor at full res)
GUIDE_EPS = 1e-2  # regularisation on RGB variance in [0, 1] units (1e-3 gave ragged edges on hair)
BAND_ROWS = 1024
AVG_R = 1  # box radius for averaging the coefficients (a wider one smears step edges over several px)
ABS_SKY_M = 10_000.0  # anything beyond 10 km is sky regardless of the percentile (keeps p99 meaningful)


def _quantiles(values: torch.Tensor, qs: tuple[float, ...]) -> list[float]:
    """Quantiles via a fixed-seed subsample of at most 1 M values (torch.quantile has a size limit)."""
    if values.numel() > QUANTILE_SAMPLES:
        g = torch.Generator(device=values.device).manual_seed(0)
        idx = torch.randint(0, values.numel(), (QUANTILE_SAMPLES,), generator=g, device=values.device)
        values = values[idx]
    out = torch.quantile(values.float(), torch.tensor(qs, device=values.device))
    return [float(v) for v in out]


def _close(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Binary closing (dilate then erode) of a [H, W] bool mask with a (2r+1)^2 square."""
    k = 2 * radius + 1
    m = mask[None, None].float()
    m = F.max_pool2d(m, k, 1, radius)
    m = -F.max_pool2d(-m, k, 1, radius)
    return m[0, 0] > 0.5


def sky_and_clamp(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """depth [h, w] metres (may contain inf) -> (finite clamped depth, sky mask, notes)."""
    notes: list[str] = []
    finite = torch.isfinite(depth) & (depth > 0) & (depth < ABS_SKY_M)
    if not finite.any():
        notes.append("no finite depth: treating the whole image as sky at 40 m")
        hi = 40.0
        return torch.full_like(depth, hi), torch.ones_like(depth, dtype=torch.bool), notes
    (p99,) = _quantiles(depth[finite], (0.99,))
    hi = SKY_FACTOR * p99
    sky = ~finite | (depth > hi)
    sky = _close(sky)
    ground = depth[finite & ~sky]
    lo = 0.9 * _quantiles(ground, (0.005,))[0] if ground.numel() else 0.0
    out = torch.where(finite, depth, torch.full_like(depth, hi)).clamp(min=max(lo, 1e-3), max=hi)
    out = torch.where(sky, torch.full_like(out, hi), out)
    notes.append(f"sky {100 * float(sky.float().mean()):.1f} % clamped to {hi:.1f} m; near clamp {lo:.2f} m")
    return out, sky, notes


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window (edge-aware normalisation) for x [1, C, h, w]."""
    return F.avg_pool2d(x, 2 * r + 1, 1, r, count_include_pad=False)


def _guided_coeffs(guide: torch.Tensor, q: torch.Tensor, r: int, eps: float):
    """Colour guided filter coefficients. guide [3, h, w], q [C, h, w] -> (a [C*3, h, w], b [C, h, w])."""
    _, h, w = guide.shape
    C = q.shape[0]
    I = guide[None]  # noqa: E741
    Q = q[None]
    mean_i = _box(I, r)[0]  # [3,h,w]
    mean_q = _box(Q, r)[0]  # [C,h,w]
    ii = torch.stack([I[0, i] * I[0, j] for i in range(3) for j in range(i, 3)])[None]
    mean_ii = _box(ii, r)[0]
    iq = torch.stack([I[0, i] * Q[0, c] for c in range(C) for i in range(3)])[None]
    mean_iq = _box(iq, r)[0].reshape(C, 3, h, w)
    # per-pixel 3x3 covariance of the guide and 3xC cross-covariance with q
    var = torch.empty(h, w, 3, 3, device=guide.device, dtype=guide.dtype)
    k = 0
    for i in range(3):
        for j in range(i, 3):
            v = mean_ii[k] - mean_i[i] * mean_i[j]
            var[..., i, j] = v
            var[..., j, i] = v
            k += 1
    var = var + eps * torch.eye(3, device=guide.device, dtype=guide.dtype)
    cov = mean_iq - mean_q[:, None] * mean_i[None]  # [C,3,h,w]
    rhs = cov.permute(2, 3, 1, 0)  # [h,w,3,C]
    a = torch.linalg.solve(var, rhs)  # [h,w,3,C]
    a = a.permute(3, 2, 0, 1)  # [C,3,h,w]
    b = mean_q - (a * mean_i[None]).sum(1)  # [C,h,w]
    # average the coefficients over the windows too (standard guided filter)
    packed = torch.cat([a.reshape(C * 3, h, w), b], 0)[None]
    packed = _box(packed, AVG_R)[0]
    return packed[: C * 3], packed[C * 3 :]


def guided_upsample(rgb: torch.Tensor, q_lr: torch.Tensor, r: int = GUIDE_RADIUS, eps: float = GUIDE_EPS):
    """Upsample q_lr [C, h, w] to the size of rgb [3, H, W] guided by rgb; returns [C, H, W] float32."""
    _, H, W = rgb.shape
    C, h, w = q_lr.shape
    guide_lr = F.adaptive_avg_pool2d(rgb[None], (h, w))[0]
    a, b = _guided_coeffs(guide_lr, q_lr, r, eps)  # low-res coefficient maps
    # The linear model can overshoot near edges: also keep the local min/max of q (over the filter window,
    # at low resolution) and clamp the result into it, so the output never leaves the neighbouring values.
    k = 2 * r + 1
    q_max = F.max_pool2d(q_lr[None], k, 1, r)[0]
    q_min = -F.max_pool2d(-q_lr[None], k, 1, r)[0]
    coeffs = torch.cat([a, b, q_min, q_max], 0)[None]  # [1, 6C, h, w]
    out = torch.empty(C, H, W, device=rgb.device, dtype=torch.float32)
    gx = ((2 * torch.arange(W, device=rgb.device, dtype=torch.float32) + 1) / W - 1)[None, :]
    for y0 in range(0, H, BAND_ROWS):
        y1 = min(H, y0 + BAND_ROWS)
        gy = ((2 * torch.arange(y0, y1, device=rgb.device, dtype=torch.float32) + 1) / H - 1)[:, None]
        grid = torch.stack([gx.expand(y1 - y0, W), gy.expand(y1 - y0, W)], dim=-1)[None]
        band = F.grid_sample(coeffs, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        a_b, b_b = band[: C * 3].reshape(C, 3, y1 - y0, W), band[C * 3 : C * 4]
        lo_b, hi_b = band[C * 4 : C * 5], band[C * 5 :]
        val = (a_b * rgb[None, :, y0:y1]).sum(1) + b_b
        out[:, y0:y1] = torch.minimum(torch.maximum(val, lo_b), hi_b)
    return out


def refine_depth(raw: RawDepth, rgb: torch.Tensor, *, model: str = "unknown") -> DepthResult:
    """RawDepth (inference resolution) + rgb [3, H, W] in [0, 1] (source resolution) -> DepthResult."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"rgb must be [3, H, W], got {tuple(rgb.shape)}")
    _, H, W = rgb.shape
    depth, sky, notes = sky_and_clamp(raw.depth.to(rgb.device, torch.float32))
    notes = list(raw.notes) + notes
    lo_val, hi_val = float(depth.min()), float(depth.max())
    if tuple(depth.shape) != (H, W):
        inv = (1.0 / depth)[None]
        q = torch.cat([inv, sky[None].float()], 0)
        up = guided_upsample(rgb.float(), q)
        depth = (1.0 / up[0].clamp(min=1.0 / hi_val, max=1.0 / lo_val)).contiguous()
        sky = up[1] > 0.5
        notes.append(f"guided upsample {raw.depth.shape[1]}x{raw.depth.shape[0]} -> {W}x{H} in inverse depth")
        depth = torch.where(sky, torch.full_like(depth, hi_val), depth)
    return DepthResult(
        depth=depth,
        f_px=raw.f_px / raw.scale,
        sky_mask=sky,
        model=model,
        infer_size=tuple(raw.infer_size),
        notes=notes,
    )
