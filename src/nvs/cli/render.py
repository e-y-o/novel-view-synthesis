"""``nvs render``: photo -> novel view at source resolution, holes marked (not filled).

Heavy imports (torch, the depth backend) happen inside ``run_render`` so ``nvs --help`` stays fast.
``run_render(args) -> RenderReport`` does the work and returns a structure (tests use it instead of parsing
stdout); ``_render`` is the argparse handler that prints the summary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from nvs.cli.errors import CliError
from nvs.licence import ACCEPT_ENV, ensure_license

HOLE_COLOURS = {"black": (0.0, 0.0, 0.0), "magenta": (1.0, 0.0, 1.0)}
CLASS_COLOURS = {
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, 0.3, 1.0),
}  # disocclusion, stretch, out of frame
DEPTH_MODELS = {"h": "depthpro2-h", "s": "depthpro2-s"}
WARN_FRACTION = 0.25


@dataclass
class RenderReport:
    output: Path
    width: int
    height: int
    intrinsics: object  # nvs.core.types.Intrinsics
    depth_model: str
    depth_cached: bool
    pose: dict[str, float]
    scale: float  # factor applied by --clamp (1.0 if none)
    coverage: dict[str, float]
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    peak_vram_gib: float | None = None
    warnings: list[str] = field(default_factory=list)
    hole_mode: str = "alpha"


def _pivot_arg(text: str) -> float | str:
    if text == "auto":
        return text
    try:
        v = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be a number of metres or 'auto'") from e
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "render",
        help="render a novel view from a single photo (holes are marked, not filled)",
        description="Render a novel view of a photo at its full resolution. Positive yaw moves the camera "
        "to the right (the scene appears to shift left), positive pitch moves it up.",
    )
    p.add_argument("image", type=Path, help="input photo (JPEG, PNG, HEIC, WebP)")
    p.add_argument("--out", type=Path, default=None, help="output .png/.jpg (default: <name>-view.png)")
    g = p.add_argument_group("pose")
    g.add_argument("--yaw", type=float, default=0.0, help="degrees, positive = camera moves right")
    g.add_argument("--pitch", type=float, default=0.0, help="degrees, positive = camera moves up")
    g.add_argument("--dolly", type=float, default=0.0, help="metres toward the pivot (negative = away)")
    g.add_argument("--pan-x", type=float, default=0.0, help="metres, camera moves right")
    g.add_argument("--pan-y", type=float, default=0.0, help="metres, camera moves down")
    g.add_argument("--pivot", type=_pivot_arg, default="auto", help="orbit pivot depth in metres or 'auto'")
    g = p.add_argument_group("intrinsics")
    g.add_argument("--f-px", type=float, default=None, help="focal length in pixels (overrides EXIF)")
    g.add_argument("--focal-35mm", type=float, default=None, help="35mm-equivalent focal length in mm")
    g = p.add_argument_group("depth")
    g.add_argument(
        "--depth-model", choices=sorted(DEPTH_MODELS), default="h", help="h (default) or s (low VRAM)"
    )
    g.add_argument("--depth-file", type=Path, default=None, help=".npy depth in metres at the image size")
    g.add_argument("--no-cache", action="store_true", help="do not read or write the depth cache")
    g.add_argument(
        "--accept-license",
        action="store_true",
        help=f"accept the Depth Pro 2 licence (or set {ACCEPT_ENV}=1)",
    )
    g = p.add_argument_group("holes")
    g.add_argument(
        "--hole", choices=["alpha", "black", "magenta"], default="alpha", help="how holes are marked"
    )
    g.add_argument("--save-mask", type=Path, default=None, help="write the feathered hole mask (8-bit PNG)")
    g.add_argument("--save-classes", type=Path, default=None, help="write a colour-coded hole class map")
    g.add_argument(
        "--fill-specks", action=argparse.BooleanOptionalAction, default=True, help="fill tiny holes"
    )
    g = p.add_argument_group("limits")
    g.add_argument("--clamp", action="store_true", help="shrink the move so holes stay under the limit")
    g.add_argument(
        "--max-hole-fraction", type=float, default=0.12, help="limit used by --clamp (default 0.12)"
    )
    g = p.add_argument_group("misc")
    g.add_argument("--save-depth-vis", type=Path, default=None, help="write a depth visualisation")
    g.add_argument("--quality", type=int, default=95, help="JPEG quality")
    g.add_argument("--timings", action="store_true", help="print a per-stage timing table")
    g.add_argument("--device", default="cuda", help="cuda (default) or cpu (needs --depth-file)")
    g.add_argument("-v", "--verbose", action="store_true", help="debug output and tracebacks")
    p.set_defaults(handler=_render)


# ------------------------------------------------------------------------------------------------ helpers
class _Timer:
    def __init__(self, device):
        self.device, self.t = device, {}

    def stage(self, name):
        return _Stage(self, name)


class _Stage:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self._sync()
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self._sync()
        self.timer.t[self.name] = self.timer.t.get(self.name, 0.0) + time.perf_counter() - self.t0

    def _sync(self):
        if str(self.timer.device).startswith("cuda"):
            import torch

            torch.cuda.synchronize()


def _save_u8(path: Path, arr) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


# ------------------------------------------------------------------------------------------------ pipeline
def run_render(args) -> RenderReport:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    from PIL import UnidentifiedImageError

    from nvs.cache import DepthCache, cache_key
    from nvs.core.camera import choose_pivot, make_camera, orbit_pose
    from nvs.core.imageio import load_image, save_image
    from nvs.core.intrinsics import resolve_intrinsics
    from nvs.depth.base import DepthError, RawDepth
    from nvs.depth.refine import refine_depth
    from nvs.depth.visualize import depth_to_color
    from nvs.render.holes import classify_holes, fill_specks
    from nvs.render.limits import estimate_coverage, limit_orbit
    from nvs.render.splat import SplatOptions, splat

    path: Path = args.image
    if not path.is_file():
        raise CliError(f"file not found: {path}")
    out = args.out or Path(f"{path.stem}-view.png")
    if out.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise CliError(f"unsupported output format {out.suffix!r}: use .png, .jpg or .jpeg")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CliError(
            "no CUDA GPU available: install an NVIDIA driver, or use --device cpu with --depth-file"
        )
    if device == "cpu" and args.depth_file is None:
        raise CliError("--device cpu needs --depth-file: the depth model requires a CUDA GPU")
    if args.max_hole_fraction <= 0:
        raise CliError("--max-hole-fraction must be positive", 2)

    timer = _Timer(device)
    warnings: list[str] = []
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    with timer.stage("load"):
        try:
            img = load_image(path, device=device)
        except UnidentifiedImageError as e:
            raise CliError(f"cannot read {path.name}: unsupported or corrupt image file") from e
        except ValueError as e:
            raise CliError(str(e)) from e
    rgb = img.rgb
    _, H, W = rgb.shape

    # --- depth backend (created lazily: only needed when EXIF has no focal length or depth is not cached)
    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            from nvs.depth.depthpro2 import DepthPro2Backend

            ensure_license(accept=args.accept_license)
            backend = DepthPro2Backend(model=DEPTH_MODELS[args.depth_model])
            backend.load()
        return backend

    estimator = None if args.depth_file is not None else (lambda: get_backend().estimate_focal(rgb))
    try:
        with timer.stage("intrinsics"):
            intr = resolve_intrinsics(
                img.meta, override_f_px=args.f_px, override_35mm=args.focal_35mm, estimator=estimator
            )
        warnings.extend(intr.warnings)

        # --- depth
        cached = False
        model_name = DEPTH_MODELS[args.depth_model]
        if args.depth_file is not None:
            try:
                d = np.load(args.depth_file, allow_pickle=False)
            except (OSError, ValueError) as e:
                raise CliError(f"cannot read depth file {args.depth_file.name}: {e}") from e
            if d.shape != (H, W):
                raise CliError(
                    f"depth file is {d.shape[1]}x{d.shape[0]} but the oriented image is {W}x{H}: "
                    "they must match"
                )
            with timer.stage("depth"):
                dt = torch.from_numpy(d.astype(np.float32)).to(device)
                depth_res = refine_depth(RawDepth(dt, intr.f_px, (H, W), 1.0), rgb, model="depth-file")
            model_name = "depth-file"
        else:
            cache = DepthCache(max_bytes=0) if args.no_cache else DepthCache()
            key = cache_key(img.sha256, model_name, "bf16", intr.f_px)
            with timer.stage("depth"):
                depth_res = cache.get(key, device=device)
                cached = depth_res is not None
                if depth_res is None:
                    be = get_backend()
                    raw = be.infer(rgb, intr.f_px)
                    depth_res = refine_depth(raw, rgb, model=model_name)
                    cache.put(key, depth_res, image_sha256=img.sha256)
        if backend is not None:  # free the VRAM the splat needs
            backend.unload()
            backend = None
            if cuda:
                torch.cuda.empty_cache()
        depth, f_px = depth_res.depth, depth_res.f_px
    except DepthError as e:
        raise CliError(str(e)) from e
    except torch.OutOfMemoryError as e:
        raise CliError(
            "GPU out of memory: close other GPU applications and retry (or use --depth-model s)"
        ) from e

    # --- pose
    pivot = choose_pivot(depth, depth_res.sky_mask) if args.pivot == "auto" else float(args.pivot)
    scale = 1.0
    yaw, pitch, dolly, pan = args.yaw, args.pitch, args.dolly, (args.pan_x, args.pan_y)
    with timer.stage("pose"):
        if args.clamp:
            lim = limit_orbit(
                depth, f_px, yaw, pitch, pivot, max_hole_fraction=args.max_hole_fraction, dolly=dolly, pan=pan
            )
            scale = lim.scale
            yaw, pitch, dolly, pan = lim.yaw_deg, lim.pitch_deg, lim.dolly, lim.pan
            if lim.limited:
                warnings.append(
                    f"move limited to {100 * scale:.0f} % (yaw {yaw:.1f}, pitch {pitch:.1f}) to keep "
                    "holes under "
                    f"{100 * args.max_hole_fraction:.0f} %"
                )
        R, t = orbit_pose(yaw, pitch, pivot, dolly, pan)
        cam = make_camera(R, t, f_px, W, H)
        if not args.clamp:
            est = estimate_coverage(depth, f_px, R, t)["total"]
            if est > WARN_FRACTION:
                warnings.append(
                    f"about {100 * est:.0f} % of the new view will be holes: "
                    "consider a smaller move or --clamp"
                )

    # --- splat, classify, compose
    with timer.stage("splat"):
        res = splat(rgb, depth, cam, options=SplatOptions(extra_channels=True))
    with timer.stage("classify"):
        hm = classify_holes(res, depth, cam)
        image = fill_specks(res.image, res.hole, hm.speck) if args.fill_specks else res.image
    hole = hm.classes > 0
    mode = args.hole
    alpha = None
    if mode == "alpha" and out.suffix.lower() in (".jpg", ".jpeg"):
        warnings.append("JPEG has no alpha channel: holes are drawn magenta instead")
        mode = "magenta"
    with timer.stage("compose"):
        if mode == "alpha":
            alpha = res.alpha * (~hole).float()
        else:
            colour = torch.tensor(HOLE_COLOURS[mode], device=image.device)[:, None, None]
            image = torch.where(hole[None], colour, image)

    comment = (
        f"nvs render yaw={yaw:g} pitch={pitch:g} dolly={dolly:g} pivot={pivot:.2f} "
        f"f_px={f_px:.1f} ({intr.source})"
    )
    with timer.stage("save"):
        save_image(out, image, alpha=alpha, source=img, quality=args.quality, comment=comment)
        if args.save_mask:
            _save_u8(args.save_mask, (hm.mask.clamp(0, 1) * 255).round().byte().cpu().numpy())
        if args.save_classes:
            cls = hm.classes.cpu()
            colours = torch.zeros(3, H, W)
            for k, c in CLASS_COLOURS.items():
                colours[:, cls == k] = torch.tensor(c)[:, None]
            _save_u8(args.save_classes, (colours.permute(1, 2, 0) * 255).round().byte().numpy())
        if args.save_depth_vis:
            vis = depth_to_color(depth, depth_res.sky_mask)
            _save_u8(args.save_depth_vis, (vis.permute(1, 2, 0) * 255).round().byte().cpu().numpy())

    return RenderReport(
        output=out,
        width=W,
        height=H,
        intrinsics=intr,
        depth_model=model_name,
        depth_cached=cached,
        pose={"yaw": yaw, "pitch": pitch, "dolly": dolly, "pan_x": pan[0], "pan_y": pan[1], "pivot": pivot},
        scale=scale,
        coverage=hm.coverage,
        timings=timer.t,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        warnings=warnings,
        hole_mode=mode,
    )


def print_report(r: RenderReport, timings: bool, file=None) -> None:
    file = file or sys.stdout
    i = r.intrinsics
    print(f"wrote {r.output} ({r.width}x{r.height}, holes: {r.hole_mode})", file=file)
    flag = " (assumed)" if i.assumed else ""
    print(f"  focal length: {i.f_px:.1f} px from {i.source}{flag}", file=file)
    print(
        f"  depth: {r.depth_model}{' (cached)' if r.depth_cached else ''}, {r.timings.get('depth', 0):.2f} s",
        file=file,
    )
    p = r.pose
    print(
        f"  pose: yaw {p['yaw']:g}, pitch {p['pitch']:g}, dolly {p['dolly']:g}, pivot {p['pivot']:.2f} m",
        file=file,
    )
    c = r.coverage
    print(
        f"  holes: {100 * c['total']:.1f} % (disocclusion {100 * c['disocclusion']:.1f}, stretch "
        f"{100 * c['stretch']:.1f}, out of frame {100 * c['out_of_frame']:.1f})",
        file=file,
    )
    if r.peak_vram_gib is not None:
        print(f"  peak VRAM: {r.peak_vram_gib:.2f} GiB", file=file)
    for w in r.warnings:
        print(f"  warning: {w}", file=file)
    if timings:
        total = sum(r.timings.values())
        print("  stage timings:", file=file)
        for k, v in r.timings.items():
            print(f"    {k:<12}{v * 1000:9.1f} ms", file=file)
        print(f"    {'total':<12}{total * 1000:9.1f} ms", file=file)


def _render(args: argparse.Namespace) -> int:
    report = run_render(args)
    print_report(report, args.timings)
    return 0
