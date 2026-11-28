"""Camera geometry shared by the CLI and (later) the browser viewer.

Conventions (OpenCV): x right, y down, z forward; the source camera is the identity at the origin.
Pixel centres sit at integer coordinates and the principal point is ((W-1)/2, (H-1)/2); fx = fy = f_px;
no distortion. A target camera is a pose (R, t) in the source frame: ``P_target = R^T (P - t)``, so ``t``
is the target camera centre (metres) and the columns of ``R`` are its right/down/forward axes.

Functions are dtype-preserving; pass float64 tensors when comparing against the golden vectors.
"""

from __future__ import annotations

import math

import torch

from nvs.core.types import CameraSpec

_PIVOT_PERCENTILE = 20.0
_DOLLY_MAX_FRACTION = 0.95
_PIVOT_MAX_SAMPLES = 1_000_000


def unproject(u, v, z, f_px: float, W: int, H: int):
    """Pixel (u, v) [px] with depth z [m] -> camera-space point (X, Y, Z) [m]. Batched (same shape)."""
    cx, cy = (W - 1) / 2, (H - 1) / 2
    return (u - cx) / f_px * z, (v - cy) / f_px * z, z


def project(X, Y, Z, f_px: float, W: int, H: int):
    """Camera-space point (X, Y, Z) [m] -> pixel (u, v) [px]. Z <= 0 gives meaningless/inf values."""
    cx, cy = (W - 1) / 2, (H - 1) / 2
    return f_px * X / Z + cx, f_px * Y / Z + cy


def transform_to_target(P: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Source-frame points P[..., 3] [m] -> target-camera frame: R^T (P - t)."""
    R = torch.as_tensor(R, dtype=P.dtype, device=P.device)
    t = torch.as_tensor(t, dtype=P.dtype, device=P.device)
    return (P - t) @ R  # row-vector form of R^T (P - t)


def orbit_pose(
    yaw_deg: float,
    pitch_deg: float,
    pivot_z: float,
    dolly: float = 0.0,
    pan: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose (R[3,3], t[3]) float32 of a camera orbiting the pivot (0, 0, pivot_z) and looking at it.

    Sign conventions (y is down in the image and the world):
    * ``yaw > 0``: the camera moves to the **right** (t_x > 0) and turns left. Near content shifts left
      in the image relative to far content. (M0's ``splat.orbit_pose`` had the opposite yaw sign.)
    * ``pitch > 0``: the camera moves **up** (t_y < 0, since +y is down) and tilts down. Near content
      shifts down (larger v) relative to far content.
    * ``dolly`` [m]: moves toward the pivot along the line of sight (positive = closer); clamped to at
      most ``0.95 * pivot_z`` so the camera never reaches the pivot. Negative moves away.
    * ``pan`` = (right, down) [m]: lateral shift along the camera's own x/y axes, applied last; the
      orientation is unchanged, so the pivot leaves the principal point when pan != 0.
    The camera has no roll (its x axis is horizontal). Yaw = pitch = dolly = pan = 0 gives the identity.
    """
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    radius = pivot_z - min(dolly, _DOLLY_MAX_FRACTION * pivot_z)
    pivot = torch.tensor([0.0, 0.0, pivot_z], dtype=torch.float64)
    offset = radius * torch.tensor(
        [math.sin(yaw) * math.cos(pitch), -math.sin(pitch), -math.cos(yaw) * math.cos(pitch)],
        dtype=torch.float64,
    )
    centre = pivot + offset
    fwd = -offset / radius
    down_ref = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    right = torch.linalg.cross(down_ref, fwd)
    right = right / right.norm()
    down = torch.linalg.cross(fwd, right)
    R = torch.stack([right, down, fwd], dim=1)
    centre = centre + pan[0] * right + pan[1] * down
    return R.to(torch.float32), centre.to(torch.float32)


def choose_pivot(depth: torch.Tensor, sky_mask: torch.Tensor | None = None, cap_m: float = 5.0) -> float:
    """Orbit pivot depth [m]: ``min(p20 of finite non-sky depth, cap_m)``; ``cap_m`` if no valid pixel.

    depth: [H, W] metres; sky_mask: [H, W] bool, True = sky. The median is deliberately not used: it is
    wrong for sky-dominated photos (M0 T06).
    """
    valid = torch.isfinite(depth) & (depth > 0)
    if sky_mask is not None:
        valid &= ~sky_mask
    vals = depth[valid].flatten()
    if vals.numel() == 0:
        return float(cap_m)
    if vals.numel() > _PIVOT_MAX_SAMPLES:  # deterministic strided subsample: a 12 MP sort would take ~250 ms
        vals = vals[:: vals.numel() // _PIVOT_MAX_SAMPLES]
    vals = vals.to(torch.float64)
    vals, _ = vals.sort()  # torch.quantile has a 16M element limit; linear interpolation as numpy
    pos = _PIVOT_PERCENTILE / 100 * (vals.numel() - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, vals.numel() - 1)
    p = float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))
    return min(p, float(cap_m))


def make_camera(R, t, f_px: float, W: int, H: int, tol: float = 1e-4) -> CameraSpec:
    """Validated CameraSpec: R must be a proper rotation (R^T R = I, det = +1), f_px > 0, sizes > 0."""
    R = torch.as_tensor(R, dtype=torch.float32)
    t = torch.as_tensor(t, dtype=torch.float32)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {tuple(R.shape)}")
    if t.shape != (3,):
        raise ValueError(f"t must have shape (3,), got {tuple(t.shape)}")
    if not (torch.isfinite(R).all() and torch.isfinite(t).all()):
        raise ValueError("R and t must be finite")
    Rd = R.to(torch.float64)
    if (Rd.T @ Rd - torch.eye(3, dtype=torch.float64)).abs().max() > tol:
        raise ValueError("R is not orthonormal")
    if torch.linalg.det(Rd) < 0:
        raise ValueError("R is a reflection (det < 0), not a rotation")
    if not f_px > 0:
        raise ValueError(f"f_px must be > 0, got {f_px}")
    if W <= 0 or H <= 0:
        raise ValueError(f"width/height must be positive, got {W}x{H}")
    return CameraSpec(R=R, t=t, f_px=float(f_px), width=int(W), height=int(H))
