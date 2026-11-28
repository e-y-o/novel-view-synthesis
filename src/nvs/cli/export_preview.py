"""``nvs export-preview``: run the analysis pipeline and write the browser preview assets to a folder.

A dev tool: frontend work gets real (or ``--stub``) data without running the server. Output files are the same
as the server's ``preview/`` directory (``color.jpg``, ``depth.f16``, ``sky.u8``, ``meta.json``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nvs.cli.errors import CliError
from nvs.licence import ACCEPT_ENV, ensure_license

DEPTH_MODELS = {"h": "depthpro2-h", "s": "depthpro2-s"}


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "export-preview",
        help="write browser preview assets (colour, depth, sky mask, meta) for a photo",
        description="Run depth estimation and write the preview asset files the web viewer loads.",
    )
    p.add_argument("image", type=Path, help="input photo (JPEG, PNG, HEIC, WebP)")
    p.add_argument("out_dir", type=Path, help="folder for color.jpg, depth.f16, sky.u8, meta.json")
    p.add_argument("--stub", action="store_true", help="fake depth, no GPU or weights (UI development)")
    p.add_argument("--depth-model", choices=sorted(DEPTH_MODELS), default="h", help="h (default) or s")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the depth cache")
    p.add_argument(
        "--accept-license",
        action="store_true",
        help=f"accept the Depth Pro 2 licence (or set {ACCEPT_ENV}=1)",
    )
    p.add_argument("--timings", action="store_true", help="print per-stage timings")
    p.add_argument("-v", "--verbose", action="store_true", help="tracebacks on errors")
    p.set_defaults(handler=_export)


def _export(args: argparse.Namespace) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import time

    import torch

    from nvs.cache import DepthCache
    from nvs.depth.base import DepthError
    from nvs.pipeline import PipelineError, analyze_image
    from nvs.preview.assets import package_preview

    path: Path = args.image
    if not path.is_file():
        raise CliError(f"file not found: {path}")
    device = "cpu" if args.stub else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise CliError("no CUDA GPU available: use --stub for fake depth, or install an NVIDIA driver")

    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            if args.stub:
                from nvs.depth.stub import StubDepthBackend

                backend = StubDepthBackend()
            else:
                from nvs.depth.depthpro2 import DepthPro2Backend

                ensure_license(accept=args.accept_license)
                backend = DepthPro2Backend(model=DEPTH_MODELS[args.depth_model])
            backend.load()
        return backend

    timings: dict[str, float] = {}

    class _Stage:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            if device == "cuda":
                torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def __exit__(self, *exc):
            if device == "cuda":
                torch.cuda.synchronize()
            timings[self.name] = timings.get(self.name, 0.0) + time.perf_counter() - self.t0

    try:
        analysis = analyze_image(
            path,
            get_backend,
            DepthCache(max_bytes=0) if args.no_cache else DepthCache(),
            model_name="stub" if args.stub else DEPTH_MODELS[args.depth_model],
            device=device,
            stage=_Stage,
        )
        with _Stage("package"):
            info = package_preview(
                analysis.image.rgb,
                analysis.depth,
                analysis.intrinsics,
                args.out_dir,
                filename=path.name,
                meta=analysis.image.meta,
                exif_bytes=analysis.image.exif_bytes,
                cached=analysis.cached,
                warnings=list(analysis.image.notes) + list(analysis.depth.notes),
            )
    except (PipelineError, DepthError) as e:
        raise CliError(str(e)) from e
    except torch.OutOfMemoryError as e:
        raise CliError("GPU out of memory: close other GPU applications and retry") from e

    print(
        f"wrote {args.out_dir} ({info['preview_width']}x{info['preview_height']} preview of "
        f"{info['width']}x{info['height']}, f={info['f_px_source']:.0f} px from "
        f"{info['intrinsics']['source']}, pivot {info['pivot_z']:.2f} m"
        f"{', depth cached' if analysis.cached else ''})"
    )
    if args.timings:
        for k, v in timings.items():
            print(f"  {k:<12}{v * 1000:9.1f} ms", file=sys.stdout)
    return 0
