"""Worker jobs on an analysed session: ``coverage``, ``limit`` and ``render``.

They use the session's depth from the depth cache, kept in a small LRU (``SessionGpuCache``) so repeated calls
do not reload it. Payload common to all three: ``{session_dir, cache_dir}``; the target camera always uses the
session's own size and focal length (never the client's). Imported by ``worker_main``; imports torch.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from nvs.cache import DepthCache, cache_key
from nvs.core.camera import make_camera
from nvs.core.imageio import LoadedImage, load_image, save_image
from nvs.core.types import DepthResult
from nvs.pipeline import PRECISION
from nvs.render.holes import classify_holes, fill_specks
from nvs.render.limits import estimate_coverage, limit_orbit
from nvs.render.splat import SplatOptions, splat
from nvs.worker.main import JobContext, JobError, handler

HOLE_COLOURS = {"black": (0.0, 0.0, 0.0), "magenta": (1.0, 0.0, 1.0)}
RESULT_JSON = ".json"


@dataclass
class SessionEntry:
    stamp: tuple  # (meta.json mtime_ns, size): a re-analysis invalidates the entry
    meta: dict
    depth: torch.Tensor  # [H, W] float32 on the job device, full resolution
    f_px: float
    source: tuple[torch.Tensor, LoadedImage] | None = None  # (rgb on CPU, image metadata) for full renders
    preview: tuple[torch.Tensor, torch.Tensor] | None = None  # (rgb, depth) at preview size


class SessionGpuCache:
    """LRU of per-session depth (on the job device) and, lazily, the decoded source image (on the CPU)."""

    def __init__(self, max_items: int = 3):
        self.max_items = max_items
        self._items: OrderedDict[str, SessionEntry] = OrderedDict()

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, session: str) -> bool:
        return session in self._items

    def get(self, session: str, session_dir: Path, cache_dir: Path | None, device: str) -> SessionEntry:
        meta_path = session_dir / "preview" / "meta.json"
        try:
            st = meta_path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError as e:
            raise JobError("not_ready", "the session has no analysed preview yet") from e
        entry = self._items.get(session)
        if entry is not None and entry.stamp == stamp:
            self._items.move_to_end(session)
            return entry
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        key = (meta.get("depth") or {}).get("cache_key") or cache_key(
            session, meta["depth"]["model"], PRECISION, meta["f_px_source"]
        )
        result: DepthResult | None = DepthCache(cache_dir).get(key, device=device) if cache_dir else None
        if result is None:
            raise JobError(
                "depth_missing",
                "the stored depth for this photo is gone: upload the photo again to analyse it",
            )
        entry = SessionEntry(stamp, meta, result.depth, float(result.f_px))
        self._items[session] = entry
        self._items.move_to_end(session)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return entry

    # -------------------------------------------------------------------------------- lazy extras
    @staticmethod
    def source(entry: SessionEntry, original: Path) -> tuple[torch.Tensor, LoadedImage]:
        if entry.source is None:
            img = load_image(original, device="cpu")
            rgb = img.rgb
            u8 = (rgb * 255).round().to(torch.uint8)
            exact = bool(torch.equal(u8.float() / 255.0, rgb))  # 16-bit sources are not multiples of 1/255
            entry.source = (u8 if exact else rgb, replace(img, rgb=torch.empty(0)))
        return entry.source

    @staticmethod
    def preview(entry: SessionEntry, preview_dir: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if entry.preview is None:
            meta = entry.meta
            pw, ph = meta["preview_width"], meta["preview_height"]
            with Image.open(preview_dir / "color.jpg") as im:
                rgb = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float() / 255
            depth = np.frombuffer((preview_dir / "depth.f16").read_bytes(), dtype="<f2").reshape(ph, pw)
            entry.preview = (rgb, torch.from_numpy(depth.astype(np.float32)))
        rgb, depth = entry.preview
        return rgb.to(device), depth.to(device)


CACHE = SessionGpuCache()


# ------------------------------------------------------------------------------------------------ helpers
def _device(ctx: JobContext) -> str:
    return "cpu" if ctx.state.config.stub else "cuda"


def _entry(ctx: JobContext, payload: dict[str, Any]) -> tuple[SessionEntry, Path]:
    session_dir = Path(payload["session_dir"])
    cache_dir = Path(payload["cache_dir"]) if payload.get("cache_dir") else None
    entry = CACHE.get(payload["session"], session_dir, cache_dir, _device(ctx))
    return entry, session_dir


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _pose(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(payload["R"], dtype=torch.float32), torch.tensor(payload["t"], dtype=torch.float32)


# -------------------------------------------------------------------------------------------------- jobs
@handler("coverage")
def coverage(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    entry, _ = _entry(ctx, payload)
    dev = _device(ctx)
    R, t = _pose(payload)
    _sync(dev)
    t0 = time.perf_counter()
    try:
        cov = estimate_coverage(entry.depth, entry.f_px, R, t)  # reduces to 0.25 MP internally
    except ValueError as e:
        raise JobError("invalid_request", str(e)) from e
    _sync(dev)
    return {"coverage": cov, "ms": round((time.perf_counter() - t0) * 1000, 2)}


@handler("limit")
def limit(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    entry, _ = _entry(ctx, payload)
    dev = _device(ctx)
    o = payload["orbit"]
    _sync(dev)
    t0 = time.perf_counter()
    try:
        res = limit_orbit(
            entry.depth,
            entry.f_px,
            float(o["yaw_deg"]),
            float(o["pitch_deg"]),
            float(o["pivot_z"]),
            max_hole_fraction=float(payload.get("max_hole_fraction") or 0.12),
            count_out_of_frame=bool(payload.get("count_out_of_frame", True)),
            dolly=float(o.get("dolly", 0.0)),
            pan=(float(o["pan"][0]), float(o["pan"][1])) if o.get("pan") else (0.0, 0.0),
        )
    except ValueError as e:
        raise JobError("invalid_request", str(e)) from e
    _sync(dev)
    return {
        "scale": res.scale,
        "yaw_deg": res.yaw_deg,
        "pitch_deg": res.pitch_deg,
        "dolly": res.dolly,
        "pan": list(res.pan),
        "hole_fraction": res.hole_fraction,
        "limited": res.limited,
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@handler("render")
def render(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    dev = _device(ctx)
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    def stage(name: str, t0: float) -> float:
        _sync(dev)
        now = time.perf_counter()
        timings[name] = round((now - t0) * 1000, 1)
        return now

    entry, session_dir = _entry(ctx, payload)
    meta = entry.meta
    size = payload.get("size", "source")
    t0 = time.perf_counter()
    if size == "preview":
        rgb, depth = CACHE.preview(entry, session_dir / "preview", dev)
        f_px, W, H = float(meta["f_px_preview"]), meta["preview_width"], meta["preview_height"]
        source = None
    else:
        original = Path(payload["original"])
        src_rgb, source = CACHE.source(entry, original)
        rgb = src_rgb.to(dev).float()
        if src_rgb.dtype == torch.uint8:
            rgb = rgb / 255.0
        depth, f_px, W, H = entry.depth, entry.f_px, meta["width"], meta["height"]
    t0 = stage("load", t0)
    ctx.check_cancel()

    R, t = _pose(payload)
    try:
        cam = make_camera(R, t, f_px, W, H)
    except ValueError as e:
        raise JobError("invalid_request", str(e)) from e
    res = splat(rgb, depth, cam, options=SplatOptions(extra_channels=True))
    t0 = stage("splat", t0)
    ctx.check_cancel()
    hm = classify_holes(res, depth, cam)
    image = fill_specks(res.image, res.hole, hm.speck) if payload.get("fill_specks", True) else res.image
    t0 = stage("classify", t0)
    hole = hm.classes > 0
    mode = payload.get("hole", "alpha")
    alpha = None
    if mode == "alpha":
        alpha = res.alpha * (~hole).float()
    else:
        colour = torch.tensor(HOLE_COLOURS[mode], device=image.device)[:, None, None]
        image = torch.where(hole[None], colour, image)
    t0 = stage("compose", t0)

    orbit = payload.get("orbit")
    if orbit:
        comment = (
            f"nvs render yaw={orbit['yaw_deg']:g} pitch={orbit['pitch_deg']:g} "
            f"dolly={orbit.get('dolly', 0):g} pivot={orbit['pivot_z']:.2f} "
            f"f_px={f_px:.1f} ({meta['intrinsics']['source']})"
        )
    else:
        t_txt = ",".join(f"{v:.3f}" for v in payload["t"])
        comment = f"nvs render t=({t_txt}) f_px={f_px:.1f} ({meta['intrinsics']['source']})"

    out_dir = session_dir / "results"
    out_dir.mkdir(exist_ok=True)
    name = payload["name"]  # e.g. "<hash>.png", validated by the server
    final = out_dir / name
    tmp = (
        out_dir / f"tmp-{os.getpid()}-{name}"
    )  # keeps the real extension: save_image picks the format from it
    ctx.check_cancel()
    try:
        save_image(tmp, image, alpha=alpha, source=source, quality=95, comment=comment)
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    stage("save", t0)
    timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
    out = {
        "name": name,
        "width": W,
        "height": H,
        "coverage": hm.coverage,
        "ms": timings,
    }
    sidecar = final.with_suffix(RESULT_JSON)
    side_tmp = sidecar.with_name(f"tmp-{os.getpid()}-{sidecar.name}")
    side_tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(side_tmp, sidecar)
    if dev.startswith("cuda") and W * H > 16_000_000:
        torch.cuda.empty_cache()  # very large images: give the memory back
    return out
