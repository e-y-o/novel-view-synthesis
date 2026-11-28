"""Depth Pro 2 backend (default: H+, bf16).

Licence / weights: Depth Pro 2 weights are under Apple's research-only licence. They are **never** part of
this repository: the model downloads them on first use into the torch hub cache (or picks them up from a
local, git-ignored ``checkpoints/`` folder). ``depth_pro_2`` is imported lazily inside ``load`` so importing
this module is cheap and needs neither the package nor the weights.

Safety (M0 T03/T04: full-resolution runs froze the PC twice): ``load`` caps this process's VRAM share before
allocating anything, images above ``max_mp`` are shrunk (the focal length is scaled with them), nothing above
``hard_max_mp`` ever runs, and an out-of-memory error is retried once at half the pixel count.
"""

from __future__ import annotations

import gc
import logging
import time
import weakref

import torch
import torch.nn.functional as F

from nvs.depth.base import DepthError, RawDepth, configure_cuda_env, plan_inference

log = logging.getLogger(__name__)

_PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
FOCAL_MAX_MP = 2.0  # the estimator does not depend on resolution; run it on a <= 2 MP copy


def _resize(rgb: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Antialiased bicubic downscale of [3, H, W] to (h, w), clamped to [0, 1]."""
    out = F.interpolate(rgb[None], size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return out[0].clamp_(0, 1)


class DepthPro2Backend:
    def __init__(
        self,
        model: str = "depthpro2-h",
        precision: str = "bf16",
        max_mp: float = 4.0,
        hard_max_mp: float = 6.0,
        vram_fraction: float = 0.92,
    ):
        if precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}")
        if max_mp > hard_max_mp:
            raise ValueError(f"max_mp {max_mp:g} exceeds hard_max_mp {hard_max_mp:g}")
        if not 0 < vram_fraction <= 1:
            raise ValueError(f"vram_fraction must be in (0, 1], got {vram_fraction}")
        self.name = model
        self.precision = precision
        self.max_mp = max_mp
        self.hard_max_mp = hard_max_mp
        self.vram_fraction = vram_fraction
        self._model = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._focal_cache: tuple[weakref.ref, float | None] | None = None

    # ---- lifecycle ----
    def load(self) -> None:
        if self._model is not None:
            return
        if not torch.cuda.is_available():
            raise DepthError("Depth Pro 2 needs a CUDA GPU, but none is available")
        configure_cuda_env()  # only effective if CUDA was not initialised yet
        torch.cuda.set_per_process_memory_fraction(self.vram_fraction)  # before any allocation
        from depth_pro_2 import create_model_and_transforms
        from depth_pro_2.model import IMAGENET_MEAN, IMAGENET_STD

        t0 = time.perf_counter()
        model, _ = create_model_and_transforms(
            self.name, device="cuda", precision=_PRECISIONS[self.precision]
        )
        model.eval()
        torch.cuda.synchronize()
        self._model = model
        self._mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device="cuda").view(3, 1, 1)
        log.info("loaded %s (%s) in %.1f s", self.name, self.precision, time.perf_counter() - t0)

    def unload(self) -> None:
        self._model = self._mean = self._std = None
        self._focal_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- helpers ----
    def _normalised(self, rgb: torch.Tensor) -> torch.Tensor:
        return (rgb.to("cuda", torch.float32) - self._mean) / self._std

    @staticmethod
    def _check_rgb(rgb: torch.Tensor) -> None:
        if rgb.ndim != 3 or rgb.shape[0] != 3 or not rgb.is_floating_point():
            raise ValueError(f"rgb must be a float [3, H, W] tensor, got {rgb.dtype} {tuple(rgb.shape)}")

    def _focal_on_copy(self, rgb: torch.Tensor) -> float | None:
        """Estimator on a <= 2 MP copy; returns f_px at the resolution of ``rgb``."""
        if self._model.fov is None:
            return None
        _, H, W = rgb.shape
        plan = plan_inference(W, H, 1.0, FOCAL_MAX_MP, self.hard_max_mp)
        small = rgb.to("cuda", torch.float32)
        if plan.scale < 1.0:
            small = _resize(small, plan.height, plan.width)
        with torch.no_grad():
            f_small = float(self._model.fov.focal_length_px(self._normalised(small)[None]))
        return f_small * (W / plan.width)

    # ---- API ----
    def estimate_focal(self, rgb: torch.Tensor) -> float | None:
        self._check_rgb(rgb)
        if self._model is None:
            self.load()
        if self._focal_cache is not None and self._focal_cache[0]() is rgb:
            return self._focal_cache[1]
        value = self._focal_on_copy(rgb)
        self._focal_cache = (weakref.ref(rgb), value)
        return value

    def _run(self, rgb: torch.Tensor, f_px: float | None, mp_limit: float) -> RawDepth:
        _, H, W = rgb.shape
        plan = plan_inference(W, H, f_px, mp_limit, self.hard_max_mp)
        notes: list[str] = []
        x = rgb.to("cuda", torch.float32)
        if plan.scale < 1.0:
            x = _resize(x, plan.height, plan.width)
            notes.append(f"downscaled {W}x{H} -> {plan.width}x{plan.height} for inference")
        f_used = plan.f_px
        if f_used is None:
            est = self.estimate_focal(rgb)
            if est is None:
                raise DepthError(f"{self.name} has no focal-length estimator and no f_px was given")
            f_used = est * plan.scale
            notes.append("focal length estimated by the depth model")
        with torch.no_grad():
            out = self._model.infer(self._normalised(x), f_px=f_used)
        depth = out["depth"].to(torch.float32)
        return RawDepth(
            depth=depth,
            f_px=float(f_used),
            infer_size=(plan.height, plan.width),
            scale=plan.scale,
            notes=notes,
        )

    def infer(self, rgb: torch.Tensor, f_px: float | None) -> RawDepth:
        self._check_rgb(rgb)
        if self._model is None:
            self.load()
        _, H, W = rgb.shape
        limit = self.max_mp
        for attempt in range(2):
            oom = False
            try:
                raw = self._run(rgb, f_px, limit)
            except torch.OutOfMemoryError:
                oom = True  # clean up outside the except block: the exception pins tensors
            if not oom:
                if attempt:
                    raw.notes.append(f"GPU out of memory; retried at {limit:.2f} MP")
                return raw
            gc.collect()
            torch.cuda.empty_cache()
            limit = min(limit, W * H / 1e6) * 0.5
        raise DepthError(
            f"GPU out of memory at {limit * 2:.1f} MP and again at {limit:.1f} MP; close other GPU apps"
        )
