"""Depth backend protocol, the raw result type and the pure resolution planner."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


class DepthError(RuntimeError):
    """Depth estimation failed in a way the user can act on (e.g. GPU out of memory)."""


def configure_cuda_env() -> None:
    """Set the allocator options that keep long GPU runs from fragmenting.

    Must run before CUDA is initialised, so the CLI entry point calls it first thing as well.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class RawDepth:
    """Depth at inference resolution (before any upsampling to source size)."""

    depth: torch.Tensor  # [h, w] float32 metres (z-depth); may contain inf (sky)
    f_px: float  # focal length used, in pixels of the inference resolution
    infer_size: tuple[int, int]  # (h, w) the model ran at
    scale: float  # inference / source linear scale (<= 1)
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class DepthBackend(Protocol):
    name: str

    def load(self) -> None: ...

    def unload(self) -> None:
        """Free VRAM (gc + empty_cache)."""

    def estimate_focal(self, rgb: torch.Tensor) -> float | None:
        """Focal length estimate in pixels at the resolution of ``rgb`` [3, H, W] in [0, 1], or None."""

    def infer(self, rgb: torch.Tensor, f_px: float | None) -> RawDepth:
        """Depth for ``rgb`` [3, H, W] in [0, 1] given ``f_px`` at that resolution (None = estimate)."""


@dataclass(frozen=True)
class InferencePlan:
    width: int
    height: int
    scale: float  # linear scale, new / old width (<= 1)
    f_px: float | None  # focal length scaled to the planned resolution


def plan_inference(
    width: int, height: int, f_px: float | None, max_mp: float, hard_max_mp: float
) -> InferencePlan:
    """Choose the inference resolution: untouched if <= max_mp megapixels, else shrunk to <= max_mp.

    The focal length is scaled by the same width ratio. ``max_mp`` above ``hard_max_mp`` is a request for
    an unsafe size and raises ValueError, as do non-positive sizes.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    if max_mp <= 0:
        raise ValueError(f"max_mp must be positive, got {max_mp}")
    if max_mp > hard_max_mp:
        raise ValueError(f"requested {max_mp:g} MP exceeds the hard limit of {hard_max_mp:g} MP")
    pixels = width * height
    if pixels <= max_mp * 1e6:
        return InferencePlan(width, height, 1.0, f_px)
    k = math.sqrt(max_mp * 1e6 / pixels)
    w, h = max(1, math.floor(width * k)), max(1, math.floor(height * k))
    while w * h > max_mp * 1e6:  # floor makes this very rare; keep the guarantee strict
        w, h = max(1, w - 1), max(1, h - 1)
    ratio = w / width
    return InferencePlan(w, h, ratio, None if f_px is None else f_px * ratio)
