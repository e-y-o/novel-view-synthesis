"""Package the preview assets the browser needs (contract: ``docs/tickets/m2/README.md``).

Files written to the output directory: ``color.jpg`` (sRGB, oriented, long side <= 2048, JPEG q92, metadata
stripped), ``depth.f16`` (little-endian float16 metres, row-major, ``preview_width x preview_height``),
``sky.u8`` (255 sky / 0 else, same size) and ``meta.json``. The directory is published atomically: files are
written to ``<dir>.tmp``, validated, and only then renamed to ``<dir>``.

Depth filter (decided by test, ``tests/test_preview_assets.py::test_depth_filter_*``). A synthetic step edge
(foreground 2 m, background 6 m, sub-pixel edge positions, four common photo sizes down to <= 2048 px) was
downscaled with each candidate and the edge was read back as "pixels closer to the foreground". Worst case
over all sizes and edge positions, error in preview pixels / number of intermediate "bridge" pixels::

    bilinear on inverse depth   0.49 px / 1 bridge     <- chosen
    area average on depth       0.67 px / 1 bridge     (exceeds 0.5 px at non-integer ratios)
    min-pool                    0.98 px / 0 bridges    (dilates the foreground by up to a pixel)

Bilinear on *inverse* depth is also exact for planar surfaces (inverse depth is linear in image space), so
ground planes and walls keep their shape. The single bridge pixel per edge is cut by the viewer's adaptive cut
test (T10). float16 stores depths from 0.3 m to the ~1 km sky clamp with a relative error of at most 2^-11
(0.049 %), well inside the 0.2 % budget, so no log-depth encoding is needed.

The sky mask is max-pooled when downscaling so thin sky gaps between branches survive.
"""

from __future__ import annotations

import json
import os
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nvs.core.camera import choose_pivot
from nvs.core.exif import ImageMeta
from nvs.core.types import DepthResult, Intrinsics
from nvs.preview.layout import SCHEMA, validate_dir

MAX_PREVIEW_SIDE = 2048
JPEG_QUALITY = 92
MAKE_TAG, MODEL_TAG = 0x010F, 0x0110
_STAT_SAMPLES = 1_000_000


class PreviewError(Exception):
    """The assets are inconsistent or could not be written."""


def preview_size(width: int, height: int) -> tuple[int, int, float]:
    """``(preview_width, preview_height, scale)`` with ``scale = preview_width / width`` (<= 1)."""
    k = min(1.0, MAX_PREVIEW_SIDE / max(width, height))
    pw, ph = max(1, round(width * k)), max(1, round(height * k))
    return pw, ph, pw / width


# ------------------------------------------------------------------------------------------- resampling
def resample_depth(depth: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """Depth [H, W] -> [ph, pw], bilinear in inverse depth (see module docstring)."""
    if tuple(depth.shape) == (ph, pw):
        return depth.float()
    inv = (1.0 / depth.float())[None, None]
    out = F.interpolate(inv, size=(ph, pw), mode="bilinear", align_corners=False)
    return (1.0 / out[0, 0]).contiguous()


def resample_sky(sky: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """Bool sky mask [H, W] -> [ph, pw]; a block is sky if any of its pixels is (max-pool)."""
    if tuple(sky.shape) == (ph, pw):
        return sky.bool()
    return F.adaptive_max_pool2d(sky.float()[None, None], (ph, pw))[0, 0] > 0.5


def resample_color(rgb: torch.Tensor, ph: int, pw: int) -> np.ndarray:
    """rgb [3, H, W] float in [0, 1] -> uint8 [ph, pw, 3], Lanczos (antialiased)."""
    arr = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if arr.shape[:2] == (ph, pw):
        return arr
    return np.asarray(Image.fromarray(arr).resize((pw, ph), Image.Resampling.LANCZOS))


# ------------------------------------------------------------------------------------------------- meta
def _clean_text(value: Any, limit: int = 64) -> str | None:
    if not isinstance(value, str):
        return None
    text = "".join(c for c in value if unicodedata.category(c)[0] != "C").strip()
    return text[:limit] or None


def exif_summary(meta: ImageMeta | None, exif_bytes: bytes | None) -> dict[str, Any]:
    """Whitelisted EXIF subset. Never GPS, serial numbers or dates."""
    make = model = None
    if exif_bytes:
        try:
            ex = Image.Exif()
            ex.load(exif_bytes)
            make, model = _clean_text(ex.get(MAKE_TAG)), _clean_text(ex.get(MODEL_TAG))
        except Exception:  # malformed EXIF is not worth failing for
            pass
    raw = meta.raw if meta is not None else {}
    f35 = raw.get("FocalLengthIn35mmFilm")
    orientation = raw.get("Orientation")
    return {
        "make": make,
        "model": model,
        "f35": float(f35) if isinstance(f35, int | float) and np.isfinite(f35) else None,
        "orientation": int(orientation) if isinstance(orientation, int | float) else None,
    }


def _depth_stats(depth: torch.Tensor, sky: torch.Tensor) -> dict[str, float | None]:
    valid = depth[~sky & torch.isfinite(depth)]
    if valid.numel() == 0:
        valid = depth[torch.isfinite(depth)].flatten()
    if valid.numel() > _STAT_SAMPLES:  # deterministic strided subsample (quantile of 12 MP is slow)
        valid = valid[:: valid.numel() // _STAT_SAMPLES + 1]
    sky_depth = float(depth[sky].max()) if bool(sky.any()) else None
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "p20": float(torch.quantile(valid.float().cpu(), 0.2)),
        "sky_depth": sky_depth,
    }


# ----------------------------------------------------------------------------------------------- package
def package_preview(
    rgb: torch.Tensor,
    depth_result: DepthResult,
    intrinsics: Intrinsics,
    out_dir: str | Path,
    *,
    filename: str = "image",
    meta: ImageMeta | None = None,
    exif_bytes: bytes | None = None,
    cached: bool = False,
    warnings: list[str] | None = None,
) -> dict:
    """Write the four assets into ``out_dir`` (atomically) and return the ``meta.json`` content.

    ``rgb`` [3, H, W] float, ``depth_result.depth`` [H, W] metres at the same size. Raises ``PreviewError``
    if the written set fails validation; ``out_dir`` is then left as it was.
    """
    out_dir = Path(out_dir)
    _, H, W = rgb.shape
    if tuple(depth_result.depth.shape) != (H, W):
        raise PreviewError(f"depth {tuple(depth_result.depth.shape)} does not match the image {H}x{W}")
    pw, ph, scale = preview_size(W, H)

    depth_p = resample_depth(depth_result.depth, ph, pw)
    sky_p = resample_sky(depth_result.sky_mask, ph, pw)
    if not bool(torch.isfinite(depth_p).all()):
        raise PreviewError("preview depth contains non-finite values")
    depth_f16 = depth_p.clamp(1e-3, 60000.0).cpu().numpy().astype("<f2")
    sky_u8 = (sky_p.cpu().numpy().astype(np.uint8)) * 255
    color = resample_color(rgb, ph, pw)

    stats = _depth_stats(depth_result.depth, depth_result.sky_mask)
    f_src = float(depth_result.f_px)
    info = {
        "schema": SCHEMA,
        "filename": filename,
        "width": W,
        "height": H,
        "preview_width": pw,
        "preview_height": ph,
        "scale": scale,
        "f_px_source": f_src,
        "f_px_preview": f_src * scale,
        "intrinsics": {
            "source": intrinsics.source,
            "assumed": bool(intrinsics.assumed),
            "warnings": list(intrinsics.warnings),
        },
        "depth": {
            **stats,
            "model": depth_result.model,
            "infer_size": [int(v) for v in depth_result.infer_size],
            "cached": bool(cached),
        },
        "pivot_z": float(choose_pivot(depth_result.depth, depth_result.sky_mask)),
        "exif": exif_summary(meta, exif_bytes),
        "warnings": list(warnings or []),
    }

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        Image.fromarray(color).save(tmp / "color.jpg", "JPEG", quality=JPEG_QUALITY, optimize=True)
        (tmp / "depth.f16").write_bytes(depth_f16.tobytes())
        (tmp / "sky.u8").write_bytes(sky_u8.tobytes())
        (tmp / "meta.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        if validate_dir(tmp) is None:
            raise PreviewError("written preview assets failed validation")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return info
