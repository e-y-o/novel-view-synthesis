"""Focal-length resolution: always returns an ``Intrinsics`` with a usable ``f_px``.

Priority order (and why):
1. Explicit override (CLI flag / UI field): the user knows best.
2. EXIF 35 mm-equivalent. Trusted over the depth model's estimator, which reads 0.56-0.79x too low at
   >= 48 mm-equivalent and is +-11 % at <= 36 mm (M0 T05). Exception: if the file is cropped (the 35 mm value
   refers to the uncropped frame) and focal-plane tags exist, the focal-plane value is better.
3. EXIF focal-plane resolution rule.
4. The depth model's estimator (injected callable, so no GPU is needed here). Never "corrected" for its bias.
5. A 26 mm-equivalent default, clearly flagged.

This module does not import torch.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from nvs.core.exif import ImageMeta, f_px_from_35mm
from nvs.core.types import Intrinsics

DEFAULT_35MM = 26.0
ESTIMATOR_DISAGREE = 0.25


def _valid(x: object) -> bool:
    return isinstance(x, int | float) and math.isfinite(x) and x > 0


def resolve_intrinsics(
    meta: ImageMeta,
    *,
    override_f_px: float | None = None,
    override_35mm: float | None = None,
    estimator: Callable[[], float | None] | None = None,
    compare_estimator: bool = False,
) -> Intrinsics:
    """Resolve the focal length [px, at oriented source resolution] following the priority order above.

    ``estimator()`` returns f_px at the source resolution (or None). It is only called when EXIF gives
    nothing, or when ``compare_estimator`` is set (then a >25 % disagreement adds an informational warning).
    ``source`` is one of "override", "35mm", "focal_plane", "estimator", "default".
    """
    warnings = list(meta.warnings)

    if override_f_px is not None:
        if not _valid(override_f_px):
            raise ValueError(f"override_f_px must be positive and finite, got {override_f_px!r}")
        return Intrinsics(float(override_f_px), "override", False, warnings)
    if override_35mm is not None:
        if not _valid(override_35mm):
            raise ValueError(f"override_35mm must be positive and finite, got {override_35mm!r}")
        return Intrinsics(
            f_px_from_35mm(float(override_35mm), meta.width, meta.height), "override", False, warnings
        )

    f_fp = meta.raw.get("f_px_focal_plane")
    cropped = any(w.startswith("cropped:") for w in meta.warnings)
    result: Intrinsics | None = None
    if meta.source == "35mm" and _valid(meta.f_px):
        if cropped and _valid(f_fp):
            warnings.append(
                "35mm-equivalent refers to the uncropped frame; using the focal-plane value instead"
            )
            result = Intrinsics(float(f_fp), "focal_plane", False, warnings)
        else:
            result = Intrinsics(float(meta.f_px), "35mm", False, warnings)
    elif meta.source == "focal_plane" and _valid(meta.f_px):
        result = Intrinsics(float(meta.f_px), "focal_plane", False, warnings)

    if result is not None:
        if compare_estimator and estimator is not None:
            est = estimator()
            if _valid(est) and abs(est / result.f_px - 1) > ESTIMATOR_DISAGREE:
                warnings.append(
                    f"info: depth-model focal estimate {est:.0f} px differs from EXIF {result.f_px:.0f} px "
                    "by more than 25 % (EXIF is used; the estimator is biased low at long focal lengths)"
                )
        return result

    est = estimator() if estimator is not None else None
    if _valid(est):
        warnings.append("focal length estimated from image")
        return Intrinsics(float(est), "estimator", True, warnings)

    warnings.append(f"no focal length available: assuming a {DEFAULT_35MM:g} mm-equivalent lens")
    return Intrinsics(f_px_from_35mm(DEFAULT_35MM, meta.width, meta.height), "default", True, warnings)
