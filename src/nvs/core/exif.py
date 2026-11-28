"""EXIF -> image metadata for reprojection: oriented size and focal length in pixels.

Rule order (design plan 4.1): FocalLengthIn35mmFilm, then FocalLength + FocalPlaneXResolution,
else f_px=None (caller falls back to the depth model's estimator, then a 26 mm default).
Prototyped and checked against real samples in M0 ticket T05.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from nvs.core import _pillow  # noqa: F401  (registers HEIF support)

FULL_FRAME_DIAG_MM = 43.267  # diagonal of a 36x24 mm frame

# Tag ids (IFD0 / Exif sub-IFD)
ORIENTATION = 0x0112
EXIF_IFD = 0x8769
FOCAL_LENGTH = 0x920A
FOCAL_LENGTH_35MM = 0xA405
FOCAL_PLANE_X_RES = 0xA20E
FOCAL_PLANE_UNIT = 0xA210
PIXEL_X_DIM = 0xA002
PIXEL_Y_DIM = 0xA003
DIGITAL_ZOOM = 0xA404

# FocalPlaneResolutionUnit -> mm per unit. 2/3 are EXIF; 4/5 are TIFF/EP values some cameras write.
UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
STANDARD_ASPECTS = (1.0, 4 / 3, 3 / 2, 16 / 9, 5 / 4, 7 / 5, 2.0)

Source = Literal["35mm", "focal_plane", "none"]


@dataclass
class ImageMeta:
    width: int  # oriented (post-rotation) size
    height: int
    f_px: float | None
    source: Source
    orientation: int | None  # original EXIF tag, None if absent
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _num(v: Any) -> float | None:
    """EXIF value -> positive finite float, else None (handles IFDRational with 0 denominator)."""
    if isinstance(v, tuple) and len(v) == 2:  # raw (num, den)
        v = v[0] / v[1] if v[1] else None
    if not isinstance(v, numbers.Real):
        return None
    f = float(v)
    return f if math.isfinite(f) and f > 0 else None


def f_px_from_35mm(f35: float, w: int, h: int) -> float:
    return f35 * math.hypot(w, h) / FULL_FRAME_DIAG_MM


def extract_image_meta(path: str | Path) -> ImageMeta:
    with Image.open(path) as im:
        stored_w, stored_h = im.size
        exif = im.getexif()
    ifd = exif.get_ifd(EXIF_IFD)
    warnings: list[str] = []
    raw: dict[str, Any] = {}

    orientation = exif.get(ORIENTATION)
    if orientation is not None:
        raw["Orientation"] = orientation
    # Same size rule as ImageOps.exif_transpose: 5-8 involve a 90 degree turn.
    swap = orientation in (5, 6, 7, 8)
    width, height = (stored_h, stored_w) if swap else (stored_w, stored_h)
    if orientation is not None and orientation not in range(1, 9):
        warnings.append(f"invalid Orientation tag {orientation!r}, ignored")

    f35 = _num(ifd.get(FOCAL_LENGTH_35MM))
    fl = _num(ifd.get(FOCAL_LENGTH))
    fpr = _num(ifd.get(FOCAL_PLANE_X_RES))
    unit = ifd.get(FOCAL_PLANE_UNIT, 2) if fpr else None
    px_x, px_y = _num(ifd.get(PIXEL_X_DIM)), _num(ifd.get(PIXEL_Y_DIM))
    zoom = _num(ifd.get(DIGITAL_ZOOM))
    for name, v in [
        ("FocalLengthIn35mmFilm", f35),
        ("FocalLength", fl),
        ("FocalPlaneXResolution", fpr),
        ("FocalPlaneResolutionUnit", unit),
        ("PixelXDimension", px_x),
        ("PixelYDimension", px_y),
        ("DigitalZoomRatio", zoom),
    ]:
        if v is not None:
            raw[name] = v

    if zoom is not None and zoom > 1:
        warnings.append(f"DigitalZoomRatio={zoom:g}: 35mm-equivalent may not include the digital crop")

    # Pixel width the focal-plane tags refer to (stored orientation), and crop/resize detection.
    ref_w = stored_w
    if px_x and px_y:
        ex_w, ex_h = int(px_x), int(px_y)
        file_aspect = stored_w / stored_h
        if (ex_w > ex_h) != (stored_w > stored_h) and abs(ex_h / ex_w - file_aspect) < 0.01 * file_aspect:
            # Tags describe the frame before rotation was baked into the pixels (common after
            # editing/downscaling: Orientation reset to 1 but PixelX/YDimension left as shot).
            warnings.append(
                f"EXIF size {ex_w}x{ex_h} is rotated relative to file {stored_w}x{stored_h} "
                "(rotation baked into pixels)"
            )
            ex_w, ex_h = ex_h, ex_w
        if (ex_w, ex_h) != (stored_w, stored_h):
            if abs(ex_w / ex_h - file_aspect) > 0.01 * file_aspect:
                warnings.append(
                    f"cropped: EXIF size {ex_w}x{ex_h} vs file {stored_w}x{stored_h}; "
                    "35mm-equivalent refers to the uncropped frame"
                )
            else:
                warnings.append(f"resized: EXIF size {ex_w}x{ex_h} vs file {stored_w}x{stored_h}")
        ref_w = ex_w

    aspect = max(width, height) / min(width, height)
    if not any(abs(aspect - a) < 0.01 * a for a in STANDARD_ASPECTS):
        warnings.append(f"non-standard aspect {aspect:.3f}: image may be cropped")

    # Rule 2 value (also computed when rule 1 wins, as a cross-check).
    f_fp = None
    if fl and fpr:
        if unit in UNIT_MM:
            res_per_mm = fpr / UNIT_MM[unit]
            sensor_w_mm = ref_w / res_per_mm
            f_fp = fl * stored_w / sensor_w_mm  # = fl * res_per_mm * stored_w / ref_w
            raw["sensor_w_mm"] = sensor_w_mm
            raw["f_px_focal_plane"] = f_fp
        else:
            warnings.append(f"unknown FocalPlaneResolutionUnit {unit!r}, focal-plane rule skipped")

    if f35:
        f_px, source = f_px_from_35mm(f35, width, height), "35mm"
        if f_fp and abs(f_fp / f_px - 1) > 0.10:
            warnings.append(f"35mm and focal-plane rules disagree: {f_px:.0f} vs {f_fp:.0f} px")
    elif f_fp:
        f_px, source = f_fp, "focal_plane"
    else:
        f_px, source = None, "none"
    return ImageMeta(width, height, f_px, source, orientation, warnings, raw)
