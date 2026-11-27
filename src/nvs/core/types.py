"""Shared data contracts for the M1 pipeline (see docs/tickets/m1/README.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch


@dataclass(slots=True)
class CameraSpec:
    """Target camera pose in the source frame (OpenCV convention): P_target = R^T (P - t)."""

    R: torch.Tensor  # (3, 3) float32 rotation
    t: torch.Tensor  # (3,) float32 translation, metres
    f_px: float  # focal length in pixels (fx = fy)
    width: int  # pixels
    height: int  # pixels


@dataclass(slots=True)
class Intrinsics:
    """Resolved focal length for the source image; always has a value."""

    f_px: float  # focal length in pixels at oriented source resolution
    source: str  # "35mm" | "focal_plane" | "none" (EXIF origin)
    assumed: bool  # True when f_px is a fallback guess
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DepthResult:
    """Metric depth for the source image at source resolution."""

    depth: torch.Tensor  # (H, W) float32, metres, finite everywhere
    f_px: float  # focal length in pixels, valid at depth's resolution
    sky_mask: torch.Tensor  # (H, W) bool, True where sky
    model: str  # name of the depth model that produced this
    infer_size: tuple[int, int]  # (H, W) the model actually ran at
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SplatResult:
    """Output of splatting to the target view."""

    image: torch.Tensor  # (C, H, W) float32 in [0, 1]
    alpha: torch.Tensor  # (H, W) float32 in [0, 1], coverage
    hole: torch.Tensor  # (H, W) bool, True where unfilled
    zbuf: torch.Tensor  # (H, W) float32, metres
    extra: torch.Tensor | None = None  # (K, H, W) optional auxiliary channels


class HoleClass(IntEnum):
    """Values of the uint8 hole-classification map."""

    VALID = 0
    DISOCCLUSION = 1
    STRETCH = 2
    OUT_OF_FRAME = 3
