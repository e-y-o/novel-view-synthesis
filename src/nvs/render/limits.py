"""Cheap coverage estimate and baseline limit for interactive use.

M0 found that large disocclusion holes (>= 25 % of a crop) are hallucinated by every infill model, so the UI
needs (a) a *coverage meter*: what fraction of the new view must be invented, and (b) a *baseline limit*: how
far the camera may move. Both run the real splat + hole classification on a heavily reduced depth map
(<= 0.25 MP by default), which takes a few milliseconds. The expected disocclusion width is roughly
``f * |t| * (1/z_near - 1/z_far)`` pixels at a depth edge, which is what the splat reproduces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nvs.core.camera import make_camera, orbit_pose
from nvs.render.holes import classify_holes
from nvs.render.splat import SplatOptions, splat

DEFAULT_MAX_HOLE_FRACTION = 0.12  # design-plan guideline
HARD_WARN_FRACTION = 0.25  # above this infill models invent objects: warn loudly
BISECTION_STEPS = 8


def _reduce_depth(depth: torch.Tensor, max_mp: float) -> tuple[torch.Tensor, int]:
    """Shrink depth [H, W] to at most ``max_mp`` megapixels; returns (reduced depth, integer factor)."""
    H, W = depth.shape
    k = max(1, math.ceil(math.sqrt(H * W / (max_mp * 1e6))))
    if k == 1:
        return depth, 1
    # Min-pooling, not averaging or nearest: the nearest surface wins at every depth edge, so thin foreground
    # structures (poles, branches) survive and keep producing disocclusions in the reduced splat.
    small = -F.max_pool2d(-depth[None, None], k, k, ceil_mode=True)[0, 0]
    return small, k


def estimate_coverage(
    depth: torch.Tensor,
    f_px: float,
    cam_R: torch.Tensor,
    cam_t: torch.Tensor,
    *,
    max_mp: float = 0.25,
) -> dict[str, float]:
    """Per-class hole fractions {"disocclusion", "stretch", "out_of_frame", "total"} for a target pose.

    depth [H, W] metres (finite; clamp sky first) and f_px are at the same resolution (source resolution is
    fine, it is reduced internally); cam_R, cam_t describe the target pose as in ``CameraSpec``.
    """
    small, k = _reduce_depth(depth.float(), max_mp)
    h, w = small.shape
    cam = make_camera(cam_R, cam_t, f_px / k, w, h)
    gray = torch.full((3, h, w), 0.5, device=small.device)
    res = splat(gray, small, cam, options=SplatOptions(extra_channels=True))
    return classify_holes(res, small, cam).coverage


@dataclass
class LimitResult:
    scale: float  # 0..1: factor applied to (yaw, pitch, dolly, pan)
    yaw_deg: float
    pitch_deg: float
    hole_fraction: float  # total hole fraction at the limited pose
    limited: bool  # True when the requested move was reduced
    dolly: float = 0.0
    pan: tuple[float, float] = (0.0, 0.0)


def limit_orbit(
    depth: torch.Tensor,
    f_px: float,
    yaw_deg: float,
    pitch_deg: float,
    pivot_z: float,
    *,
    max_hole_fraction: float = DEFAULT_MAX_HOLE_FRACTION,
    dolly: float = 0.0,
    pan: tuple[float, float] = (0.0, 0.0),
    max_mp: float = 0.25,
    count_out_of_frame: bool = True,
) -> LimitResult:
    """Largest ``scale`` in [0, 1] such that ``scale * (yaw, pitch, dolly, pan)`` keeps the total hole
    fraction at or below ``max_hole_fraction``. Out-of-frame holes count unless ``count_out_of_frame`` is
    False (a telephoto photo reaches the limit through out-of-frame area alone, which infill could not fill
    well anyway); ``hole_fraction`` of the result is the counted fraction.

    Bisection with at most 8 splats. The hole fraction grows with the move on typical scenes (see the
    monotonicity tests); if it ever does not, bisection still returns a scale whose fraction was measured to
    be within the limit.
    """
    small, k = _reduce_depth(depth.float(), max_mp)  # reduce once, reuse for every probe
    h, w = small.shape
    gray = torch.full((3, h, w), 0.5, device=small.device)
    f_small = f_px / k

    def fraction(scale: float) -> float:
        R, t = orbit_pose(
            yaw_deg * scale, pitch_deg * scale, pivot_z, dolly * scale, (pan[0] * scale, pan[1] * scale)
        )
        cam = make_camera(R, t, f_small, w, h)
        res = splat(gray, small, cam, options=SplatOptions(extra_channels=True))
        cov = classify_holes(res, small, cam).coverage
        return cov["total"] if count_out_of_frame else cov["total"] - cov["out_of_frame"]

    def result(scale: float, frac: float) -> LimitResult:
        return LimitResult(
            scale=scale,
            yaw_deg=yaw_deg * scale,
            pitch_deg=pitch_deg * scale,
            hole_fraction=frac,
            limited=scale < 1.0,
            dolly=dolly * scale,
            pan=(pan[0] * scale, pan[1] * scale),
        )

    full = fraction(1.0)
    if full <= max_hole_fraction:
        return result(1.0, full)
    lo, hi, best_frac = 0.0, 1.0, 0.0
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        f = fraction(mid)
        if f <= max_hole_fraction:
            lo, best_frac = mid, f
        else:
            hi = mid
    return result(lo, best_frac)
