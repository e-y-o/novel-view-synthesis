"""Depth visualisation helpers (no matplotlib)."""

from __future__ import annotations

import torch


def _turbo_lut(n: int = 256) -> torch.Tensor:
    """Turbo-like perceptual colormap [n, 3] from a polynomial approximation (Mikhailov, Apache-2.0)."""
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    v4 = torch.stack([torch.ones_like(x), x, x**2, x**3], dim=1)
    v2 = torch.stack([x**4, x**5], dim=1)
    red = torch.tensor([0.13572138, 4.61539260, -42.66032258, 132.13108234], dtype=torch.float64)
    red2 = torch.tensor([-152.94239396, 59.28637943], dtype=torch.float64)
    green = torch.tensor([0.09140261, 2.19418839, 4.84296658, -14.18503333], dtype=torch.float64)
    green2 = torch.tensor([4.27729857, 2.82956604], dtype=torch.float64)
    blue = torch.tensor([0.10667330, 12.64194608, -60.58204836, 110.36276771], dtype=torch.float64)
    blue2 = torch.tensor([-89.90310912, 27.34824973], dtype=torch.float64)
    rgb = torch.stack([v4 @ red + v2 @ red2, v4 @ green + v2 @ green2, v4 @ blue + v2 @ blue2], dim=1)
    return rgb.clamp(0, 1).float()


_LUT = _turbo_lut()


def depth_to_color(
    depth: torch.Tensor, sky_mask: torch.Tensor | None = None, clip: tuple[float, float] = (2.0, 98.0)
) -> torch.Tensor:
    """depth [H, W] metres -> RGB [3, H, W] in [0, 1]: inverse depth, percentile-clipped, near = warm.

    Sky pixels (sky_mask True) are drawn black and excluded from the percentiles.
    """
    inv = 1.0 / depth.float().clamp_min(1e-6)
    valid = torch.isfinite(inv)
    if sky_mask is not None:
        valid &= ~sky_mask
    vals = inv[valid]
    if vals.numel() > 1_000_000:
        vals = vals[torch.randint(0, vals.numel(), (1_000_000,), device=vals.device)]
    if vals.numel() == 0:
        return torch.zeros(3, *depth.shape, device=depth.device)
    lo, hi = (
        float(v)
        for v in torch.quantile(vals, torch.tensor([clip[0] / 100, clip[1] / 100], device=vals.device))
    )
    t = ((inv - lo) / max(hi - lo, 1e-9)).clamp(0, 1)
    idx = (t * (_LUT.shape[0] - 1)).round().long()
    rgb = _LUT.to(depth.device)[idx].permute(2, 0, 1)
    if sky_mask is not None:
        rgb = rgb * (~sky_mask).float()
    return rgb.contiguous()
