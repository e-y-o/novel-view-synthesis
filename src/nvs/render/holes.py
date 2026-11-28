"""Hole classification and mask post-processing for the splatted view.

Holes have different causes and are handled differently later (preview fill in M3, generative infill in M4):

* 1 **disocclusion**: background that was hidden behind a foreground object;
* 2 **stretch**: a surface seen at a grazing angle, stretched past what the splat footprint can cover;
* 3 **out-of-frame**: the new camera sees beyond the border of the source photo.

Precedence when rules conflict: out-of-frame > disocclusion > stretch; anything unlabelled defaults to
disocclusion (the safe choice for infill).

How it works
------------
* ``splat(..., SplatOptions(extra_channels=True))`` carries two per-source-pixel flags through the warp:
  *discontinuity* (a depth edge in the source) and *stretch* (footprint beyond ``r_max``). Around each hole
  pixel (9x9 window) we look at how much warped flag lies next to it: discontinuity -> class 1, else stretch
  -> class 2, else class 1.
* Out-of-frame: the source image border (``2 (W + H)`` pixels with their depths) is projected into the
  target and the resulting closed polyline is rasterised with an even-odd scanline fill; hole pixels
  *outside* it are class 3. Limitation: the border pixels have different depths, so the projected polygon is
  ragged where the border crosses depth edges, and at extreme angles (border points behind the camera, fewer
  than 3 valid vertices) no out-of-frame class is assigned.
* Specks: connected hole components of fewer than 4 pixels, completely surrounded by valid pixels (a
  component touching the image border is not a speck), are not holes. ``find_specks`` locates them and
  ``fill_specks`` paints them with the average of their nearest valid neighbours. They are excluded from the
  classes and the coverage.
* Mask: holes are dilated by 3 px (5 px along disocclusions, i.e. +2) and the edge is feathered outward with a
  2 px Gaussian; the hole interior is exactly 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nvs.core.types import CameraSpec, HoleClass, SplatResult

CLASS_NAMES = {
    HoleClass.DISOCCLUSION: "disocclusion",
    HoleClass.STRETCH: "stretch",
    HoleClass.OUT_OF_FRAME: "out_of_frame",
}


@dataclass
class HoleOptions:
    flag_window: int = 9  # window (px) in which warped flags next to a hole are measured
    flag_threshold: float = 0.02  # minimum flag density in that window to decide the class
    speck_max: int = 3  # components of at most this many pixels are specks
    dilate: int = 3  # mask dilation radius (px)
    dilate_disocclusion_extra: int = 2  # extra radius along disocclusions
    feather_sigma: float = 2.0  # Gaussian feather (px)


@dataclass
class HoleMap:
    classes: torch.Tensor  # [H, W] uint8, values of HoleClass (0 = valid)
    mask: torch.Tensor  # [H, W] float32 in [0, 1]: dilated, feathered, interior 1
    coverage: dict[str, float]  # fraction of pixels per class and "total"
    speck: torch.Tensor | None = None  # [H, W] bool: tiny holes that were cleaned up


# ---------------------------------------------------------------- specks
def find_specks(hole: torch.Tensor, max_size: int = 3) -> torch.Tensor:
    """bool [H, W]: pixels of connected (8-neighbour) hole components with at most ``max_size`` pixels.

    A component has more than ``max_size`` pixels iff a breadth-first search from any of its pixels reaches
    that many pixels within ``max_size`` steps, so only a (2 max_size + 1)^2 window per candidate is needed.
    The image border counts as non-valid (a hole touching it is not a speck).
    """
    H, W = hole.shape
    out = torch.zeros_like(hole)
    k = max_size
    # candidates: holes with few holes around them (every pixel of a tiny component has at most max_size
    # hole pixels in its 5x5 window)
    count5 = F.avg_pool2d(hole.float()[None, None], 5, 1, 2, count_include_pad=True)[0, 0] * 25
    cand = (hole & (count5 <= max_size + 0.5)).nonzero()
    if cand.numel() == 0:
        return out
    padded = F.pad(hole[None, None].float(), (k, k, k, k), value=1.0)[0, 0] > 0.5  # outside = not valid
    off = torch.arange(-k, k + 1, device=hole.device)
    ys = cand[:, 0, None, None] + k + off[None, :, None]
    xs = cand[:, 1, None, None] + k + off[None, None, :]
    windows = padded[ys, xs]  # [n, win, win]
    reach = torch.zeros_like(windows)
    reach[:, k, k] = True
    for _ in range(k):
        reach = (F.max_pool2d(reach.float()[:, None], 3, 1, 1)[:, 0] > 0.5) & windows
    small = reach.flatten(1).sum(1) <= max_size
    out[cand[small, 0], cand[small, 1]] = True
    return out


def fill_specks(image: torch.Tensor, hole: torch.Tensor, speck: torch.Tensor | None = None) -> torch.Tensor:
    """Paint speck pixels of ``image`` [C, H, W] with the mean of their nearest valid neighbours.

    ``hole`` [H, W] bool is the splat hole mask; ``speck`` defaults to ``find_specks(hole)``. Returns a new
    image; only speck pixels change.
    """
    speck = find_specks(hole) if speck is None else speck
    out = image.clone()
    todo = speck.nonzero()
    if todo.numel() == 0:
        return out
    C, H, W = image.shape
    valid = ~hole
    ys, xs = todo[:, 0], todo[:, 1]
    for _ in range(4):  # a speck has at most 3 pixels, so 3 rings are enough; one spare
        if ys.numel() == 0:
            break
        acc = torch.zeros(ys.numel(), C, device=image.device, dtype=image.dtype)
        cnt = torch.zeros(ys.numel(), device=image.device, dtype=image.dtype)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y2, x2 = ys + dy, xs + dx
                inside = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
                y2c, x2c = y2.clamp(0, H - 1), x2.clamp(0, W - 1)
                ok = inside & valid[y2c, x2c]
                acc += out[:, y2c, x2c].T * ok[:, None]
                cnt += ok
        filled = cnt > 0
        out[:, ys[filled], xs[filled]] = (acc[filled] / cnt[filled, None]).T
        valid[ys[filled], xs[filled]] = True
        ys, xs = ys[~filled], xs[~filled]
    return out


# ---------------------------------------------------------------- out of frame
def _border_polyline(depth: torch.Tensor, cam: CameraSpec, f_src: float):
    """Target-space polygon (x, y) of the projected source border, in border order; drops invalid points."""
    H, W = depth.shape
    dev = depth.device
    top = torch.stack([torch.arange(W), torch.zeros(W)], 1)
    right = torch.stack([torch.full((H - 1,), W - 1), torch.arange(1, H)], 1)
    bottom = torch.stack([torch.arange(W - 2, -1, -1), torch.full((W - 1,), H - 1)], 1)
    left = torch.stack([torch.zeros(H - 2), torch.arange(H - 2, 0, -1)], 1)
    uv = torch.cat([top, right, bottom, left]).to(dev, torch.float32)
    u, v = uv[:, 0], uv[:, 1]
    z = depth[v.long(), u.long()]
    R, t = cam.R.to(dev, torch.float32), cam.t.to(dev, torch.float32)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    P = torch.stack([(u - cx) / f_src * z, (v - cy) / f_src * z, z], 1)
    Q = (P - t) @ R
    ok = Q[:, 2] > 1e-3
    x = cam.f_px * Q[:, 0] / Q[:, 2].clamp_min(1e-3) + (cam.width - 1) / 2
    y = cam.f_px * Q[:, 1] / Q[:, 2].clamp_min(1e-3) + (cam.height - 1) / 2
    pts = torch.stack([x, y], 1)[ok]
    return pts if pts.shape[0] >= 3 and bool(torch.isfinite(pts).all()) else None


def inside_source_frame(
    depth: torch.Tensor, cam: CameraSpec, f_src: float | None = None
) -> torch.Tensor | None:
    """bool [H, W] (target size): pixels inside the projected source-image border, or None if undefined."""
    f_src = cam.f_px if f_src is None else f_src
    pts = _border_polyline(depth, cam, f_src)
    if pts is None:
        return None
    H, W = cam.height, cam.width
    dev = pts.device
    p0, p1 = pts, torch.roll(pts, -1, 0)
    x0, y0, x1, y1 = p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1]
    lo = torch.ceil(torch.minimum(y0, y1)).clamp(0, H).long()
    hi = (
        torch.ceil(torch.maximum(y0, y1)).clamp(0, H).long()
    )  # exclusive; half-open rule on pixel-centre rows
    counts = ((hi - lo) * (y0 != y1)).clamp_min(0)
    total = int(counts.sum())
    if total == 0:
        return torch.zeros(H, W, dtype=torch.bool, device=dev)
    edge = torch.repeat_interleave(torch.arange(pts.shape[0], device=dev), counts)
    start = torch.cumsum(counts, 0) - counts
    rows = lo[edge] + (torch.arange(total, device=dev) - start[edge])
    xc = x0[edge] + (rows.float() - y0[edge]) * (x1[edge] - x0[edge]) / (y1[edge] - y0[edge])
    col = torch.ceil(xc).clamp(0, W).long()  # first pixel centre at or right of the crossing
    diff = torch.bincount(rows * (W + 1) + col, minlength=H * (W + 1)).reshape(H, W + 1)
    return (diff.cumsum(1)[:, :W] % 2) == 1


def _flag_density(flag: torch.Tensor, scale: int, win: int) -> torch.Tensor:
    """Mean of ``flag`` [H, W] over a window of win x win cells of size ``scale`` px, as a full-size map."""
    H, W = flag.shape
    x = flag[None, None]
    if scale > 1:
        x = F.avg_pool2d(x, scale, scale, ceil_mode=True)
    x = F.avg_pool2d(x, win, 1, win // 2, count_include_pad=True)
    if scale > 1:
        x = F.interpolate(x, scale_factor=scale, mode="nearest")[..., :H, :W]
    return x[0, 0]


# ---------------------------------------------------------------- mask
def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    return F.max_pool2d(m[None, None], 2 * r + 1, 1, r)[0, 0]


def make_mask(classes: torch.Tensor, options: HoleOptions) -> torch.Tensor:
    """Dilate (+2 px more along disocclusions) and feather outward; the hole interior stays exactly 1."""
    holes = (classes > 0).float()
    grown = _dilate(holes, options.dilate)
    disoc = (classes == HoleClass.DISOCCLUSION).float()
    grown = torch.maximum(grown, _dilate(disoc, options.dilate + options.dilate_disocclusion_extra))
    if options.feather_sigma <= 0:
        return grown
    k = _gaussian_kernel(options.feather_sigma, grown.device)
    r = (k.numel() - 1) // 2
    x = grown[None, None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return torch.where(grown > 0.5, torch.ones_like(grown), x[0, 0])


# ---------------------------------------------------------------- main entry
def classify_holes(
    res: SplatResult, depth: torch.Tensor, cam: CameraSpec, options: HoleOptions | None = None
) -> HoleMap:
    """Classify the holes of ``res`` (a splat made with ``SplatOptions(extra_channels=True)``).

    ``depth`` is the source depth [H, W] used for the splat, ``cam`` the target camera (the source shares its
    intrinsics and size).
    """
    opts = options or HoleOptions()
    if res.extra is None or res.extra.shape[0] < 2:
        raise ValueError("classify_holes needs a splat made with SplatOptions(extra_channels=True)")
    hole = res.hole
    H, W = hole.shape
    speck = find_specks(hole, opts.speck_max)
    real = hole & ~speck

    classes = torch.full((H, W), int(HoleClass.DISOCCLUSION), dtype=torch.uint8, device=hole.device)
    todo = real.clone()
    # Look for warped flags next to each hole pixel, first in a small window, then (for the interior of big
    # holes) in windows 4x and 16x larger, evaluated on pooled maps. Disocclusion wins over stretch.
    for scale, thr in (
        (1, opts.flag_threshold),
        (4, opts.flag_threshold / 4),
        (16, opts.flag_threshold / 16),
    ):
        disc_d = _flag_density(res.extra[0], scale, opts.flag_window)
        str_d = _flag_density(res.extra[1], scale, opts.flag_window)
        is_disc, is_str = disc_d > thr, (str_d > thr) & ~(disc_d > thr)
        classes[todo & is_str] = int(HoleClass.STRETCH)
        todo &= ~(is_disc | is_str)
        if not bool(todo.any()):
            break
    inside = inside_source_frame(depth, cam)
    if inside is not None:
        classes[~inside] = int(HoleClass.OUT_OF_FRAME)
    classes = torch.where(real, classes, torch.zeros_like(classes))

    total = float(real.float().mean())
    coverage = {name: float((classes == int(c)).float().mean()) for c, name in CLASS_NAMES.items()}
    coverage["total"] = total
    return HoleMap(classes=classes, mask=make_mask(classes, opts), coverage=coverage, speck=speck)
