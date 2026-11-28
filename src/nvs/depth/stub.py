"""Deterministic fake depth backend: no GPU, no weights. For UI development and tests.

The scene is a tilted ground plane (near at the bottom of the image, far at the top) with a near ellipse in
the middle. It depends only on the image size, never on its content.
"""

from __future__ import annotations

import torch

from nvs.depth.base import RawDepth

NEAR_PLANE_M = 2.0  # ground at the bottom row
FAR_PLANE_M = 12.0  # ground at the top row
ELLIPSE_M = 1.5


class StubDepthBackend:
    name = "stub"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @staticmethod
    def _check_rgb(rgb: torch.Tensor) -> None:
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"expected rgb of shape [3, H, W], got {tuple(rgb.shape)}")

    def estimate_focal(self, rgb: torch.Tensor) -> float | None:
        self._check_rgb(rgb)
        return 0.9 * max(rgb.shape[-2:])

    def infer(self, rgb: torch.Tensor, f_px: float | None) -> RawDepth:
        self._check_rgb(rgb)
        h, w = int(rgb.shape[1]), int(rgb.shape[2])
        v = (torch.arange(h, dtype=torch.float32) + 0.5) / h  # 0 top .. 1 bottom
        depth = (NEAR_PLANE_M + (FAR_PLANE_M - NEAR_PLANE_M) * (1.0 - v)).view(h, 1).expand(h, w).clone()
        u = (torch.arange(w, dtype=torch.float32) + 0.5) / w
        inside = ((u.view(1, w) - 0.5) / 0.2) ** 2 + ((v.view(h, 1) - 0.5) / 0.25) ** 2 <= 1.0
        depth[inside] = ELLIPSE_M
        return RawDepth(
            depth=depth,
            f_px=float(f_px) if f_px is not None else 0.9 * max(h, w),
            infer_size=(h, w),
            scale=1.0,
            notes=["stub depth: synthetic scene, not an estimate"],
        )
