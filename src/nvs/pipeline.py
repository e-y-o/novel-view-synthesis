"""Steps shared by ``nvs render``, ``nvs export-preview`` and the server's ``analyze`` job.

``analyze_image``: load (oriented, sRGB) -> intrinsics -> depth (cache hit skips inference) -> refine. The
depth backend is only touched when needed: the focal-length estimator runs only when EXIF has no focal length,
and inference only on a cache miss. ``backend`` may be a ready backend or a lazy zero-argument factory.

Progress phases (``progress(phase, pct)``): ``decode`` -> ``intrinsics`` -> ``depth`` (the server adds
``package``). ``stage(name)`` is an optional context-manager factory so callers can time the steps.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import UnidentifiedImageError

from nvs.cache import DepthCache, cache_key
from nvs.core.imageio import LoadedImage, load_image
from nvs.core.intrinsics import resolve_intrinsics
from nvs.core.types import DepthResult, Intrinsics
from nvs.depth.base import DepthBackend, RawDepth
from nvs.depth.refine import refine_depth

PRECISION = "bf16"
Progress = Callable[[str, float], None]


class PipelineError(Exception):
    """A user-readable failure (bad image, mismatching depth file)."""


@dataclass
class AnalysisResult:
    depth: DepthResult
    intrinsics: Intrinsics
    image: LoadedImage
    model: str
    cached: bool


def analyze_image(
    path: str | Path,
    backend: DepthBackend | Callable[[], DepthBackend] | None,
    cache: DepthCache | None,
    *,
    model_name: str | None = None,
    device: str = "cuda",
    override_f_px: float | None = None,
    override_35mm: float | None = None,
    depth_array: np.ndarray | None = None,
    progress: Progress | None = None,
    stage: Callable[[str], AbstractContextManager[Any]] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> AnalysisResult:
    """Analyse one photo. ``depth_array`` (metres, oriented image size) replaces the depth model."""
    stage = stage or (lambda name: nullcontext())
    report = progress or (lambda phase, pct: None)
    cancel = check_cancel or (lambda: None)

    get_backend_fn: Callable[[], DepthBackend] | None
    if backend is None:
        get_backend_fn = None
    elif callable(backend) and not isinstance(backend, DepthBackend):
        get_backend_fn = backend
    else:
        get_backend_fn = lambda: backend  # noqa: E731

    def get_backend() -> DepthBackend:
        if get_backend_fn is None:
            raise PipelineError("a depth backend is required but none was provided")
        return get_backend_fn()

    report("decode", 0.0)
    with stage("load"):
        try:
            img = load_image(path, device=device)
        except (UnidentifiedImageError, OSError) as e:  # OSError: truncated or corrupt data
            raise PipelineError(f"cannot read {Path(path).name}: unsupported or corrupt image file") from e
        except ValueError as e:
            raise PipelineError(str(e)) from e
    rgb = img.rgb
    _, H, W = rgb.shape
    cancel()

    report("intrinsics", 15.0)
    estimator = None if depth_array is not None else (lambda: get_backend().estimate_focal(rgb))
    with stage("intrinsics"):
        intr = resolve_intrinsics(
            img.meta, override_f_px=override_f_px, override_35mm=override_35mm, estimator=estimator
        )
    cancel()

    report("depth", 25.0)
    cached = False
    if depth_array is not None:
        d = depth_array
        if d.shape != (H, W):
            raise PipelineError(
                f"depth file is {d.shape[1]}x{d.shape[0]} but the oriented image is {W}x{H}: they must match"
            )
        with stage("depth"):
            dt = torch.from_numpy(d.astype(np.float32)).to(device)
            depth_res = refine_depth(RawDepth(dt, intr.f_px, (H, W), 1.0), rgb, model="depth-file")
        model = "depth-file"
    else:
        if model_name is None:
            raise PipelineError("model_name is required when no depth array is given")
        model = model_name
        cache = cache if cache is not None else DepthCache(max_bytes=0)
        key = cache_key(img.sha256, model, PRECISION, intr.f_px)
        with stage("depth"):
            depth_res = cache.get(key, device=device)
            cached = depth_res is not None
            if depth_res is None:
                raw = get_backend().infer(rgb, intr.f_px)
                cancel()
                depth_res = refine_depth(raw, rgb, model=model)
                cache.put(key, depth_res, image_sha256=img.sha256)
    return AnalysisResult(depth=depth_res, intrinsics=intr, image=img, model=model, cached=cached)
