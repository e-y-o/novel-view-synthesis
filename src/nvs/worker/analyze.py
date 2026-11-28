"""Worker job ``analyze``: photo -> depth (cached) -> preview assets in ``<session>/preview``.

Payload: ``{path, out_dir, filename, cache_dir?, model?}``. Progress events carry ``phase`` (``decode``,
``intrinsics``, ``depth``, ``package``) and an overall ``pct``. Errors use stable codes:
``unsupported_image``, ``gpu_oom``, ``depth_failed``; a missing licence acceptance surfaces as
``licence_not_accepted`` from the backend.
Imported by ``worker_main`` (it registers the handler); it imports torch: the web process must not.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from nvs.cache import DepthCache
from nvs.depth.base import DepthError
from nvs.pipeline import PipelineError, analyze_image
from nvs.preview.assets import PreviewError, package_preview
from nvs.worker.main import JobContext, JobError, handler

DEFAULT_MODEL = "depthpro2-h"


class _Timings:
    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.t: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        if self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda:
                torch.cuda.synchronize()
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0


@handler("analyze")
def analyze(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    stub = ctx.state.config.stub
    device = "cpu" if stub else "cuda"
    timings = _Timings(cuda=not stub)
    model = payload.get("model") or ctx.state.config.depth_model
    cache = DepthCache(Path(payload["cache_dir"])) if payload.get("cache_dir") else DepthCache()

    def progress(phase: str, pct: float) -> None:
        ctx.progress(pct, phase=phase)

    try:
        analysis = analyze_image(
            payload["path"],
            ctx.backend,  # lazy: only touched when EXIF lacks a focal length or the depth is not cached
            cache,
            model_name=model if not stub else "stub",
            device=device,
            progress=progress,
            stage=timings.stage,
            check_cancel=ctx.check_cancel,
        )
        ctx.check_cancel()
        progress("package", 90.0)
        with timings.stage("package"):
            info = package_preview(
                analysis.image.rgb,
                analysis.depth,
                analysis.intrinsics,
                payload["out_dir"],
                filename=payload.get("filename", "image"),
                meta=analysis.image.meta,
                exif_bytes=analysis.image.exif_bytes,
                cached=analysis.cached,
                cache_key=analysis.cache_key,
                warnings=list(analysis.image.notes) + list(analysis.depth.notes),
            )
    except PipelineError as e:
        raise JobError("unsupported_image", str(e)) from e
    except torch.OutOfMemoryError as e:
        raise JobError(
            "gpu_oom", "The GPU ran out of memory: close other GPU applications and try again."
        ) from e
    except DepthError as e:
        raise JobError("depth_failed", str(e)) from e
    except PreviewError as e:
        raise JobError("depth_failed", f"could not package the preview: {e}") from e
    finally:
        if not stub:
            torch.cuda.empty_cache()  # leave room for the splat / next job
    progress("package", 100.0)
    return {
        "width": info["width"],
        "height": info["height"],
        "cached": analysis.cached,
        "timings_ms": {k: round(v * 1000, 1) for k, v in timings.t.items()},
    }
