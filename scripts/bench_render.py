"""Benchmark the render pipeline stages at 2 / 12 / 24 MP on synthetic (resampled) data.

    uv run python scripts/bench_render.py [--sizes 2 12 24] [--runs 7] [--with-depth] [--out data/bench]

Timing rules (M0): CUDA is synchronised before reading the clock, 2 warm-up runs, then ``--runs`` timed runs;
the table shows the median (min-max in the JSON). Stages: load (JPEG decode to GPU tensor), refine (guided
upsample from a <= 4 MP depth), splat, classify, coverage (0.25 MP estimate), save (PNG, RGBA). With
``--with-depth`` the Depth Pro 2 backend is also timed: cold (model load) and warm (4 MP inference).
Writes ``data/bench/bench-<timestamp>.json`` and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def timeit(fn, runs: int, warmup: int = 2):
    import torch

    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}


def synthetic(mp: float, device: str):
    """Smooth colour image + depth with a foreground object, at ``mp`` megapixels (4:3)."""
    import torch

    h = round(math.sqrt(mp * 1e6 * 3 / 4))
    w = round(h * 4 / 3)
    ys = torch.linspace(0, 1, h, device=device)[:, None].expand(h, w)
    xs = torch.linspace(0, 1, w, device=device)[None, :].expand(h, w)
    g = torch.Generator(device=device).manual_seed(0)
    rgb = torch.stack([xs, ys, (xs + ys) / 2]) * 0.8 + 0.2 * torch.rand(3, h, w, device=device, generator=g)
    depth = 2.0 + 6.0 * ys + 0.5 * torch.sin(xs * 30)
    depth[h // 3 : 2 * h // 3, w // 3 : w // 2] = 1.5
    return rgb.clamp(0, 1), depth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=float, nargs="+", default=[2, 12, 24], help="megapixels")
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--with-depth", action="store_true", help="also time Depth Pro 2 (needs weights)")
    ap.add_argument("--out", type=Path, default=Path("data/bench"))
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("error: a CUDA GPU is required for the benchmark")
        return 1
    from PIL import Image

    from nvs.core.camera import make_camera, orbit_pose
    from nvs.core.imageio import load_image, save_image
    from nvs.depth.base import RawDepth
    from nvs.depth.refine import refine_depth
    from nvs.render.holes import classify_holes, fill_specks
    from nvs.render.limits import estimate_coverage
    from nvs.render.splat import SplatOptions, splat

    dev = "cuda"
    results: dict[str, dict] = {}
    tmp = Path(tempfile.mkdtemp())
    for mp in args.sizes:
        rgb, depth = synthetic(mp, dev)
        _, h, w = rgb.shape
        f = 0.9 * max(w, h)
        R, t = orbit_pose(8, 0, 3.0)
        cam = make_camera(R, t, f, w, h)
        opts = SplatOptions(extra_channels=True)
        jpg = tmp / f"s{mp:g}.jpg"
        Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(jpg, quality=92)
        k = max(1, math.ceil(math.sqrt(w * h / 4e6)))
        raw_depth = depth[::k, ::k].contiguous()
        raw = RawDepth(raw_depth, f / k, tuple(raw_depth.shape), 1 / k)
        res = splat(rgb, depth, cam, options=opts)
        hm = classify_holes(res, depth, cam)
        stages = {
            "load": timeit(lambda: load_image(jpg, device=dev), args.runs),
            "refine": timeit(lambda: refine_depth(raw, rgb), args.runs) if k > 1 else {"median_ms": 0.0},
            "splat": timeit(lambda: splat(rgb, depth, cam, options=opts), args.runs),
            "classify": timeit(
                lambda: (classify_holes(res, depth, cam), fill_specks(res.image, res.hole)), args.runs
            ),
            "coverage": timeit(lambda: estimate_coverage(depth, f, R, t), args.runs),
            "save": timeit(
                lambda: save_image(tmp / "out.png", res.image, alpha=res.alpha * (hm.classes == 0).float()),
                min(args.runs, 3),
                warmup=1,
            ),
        }
        stages["total_warm"] = {
            "median_ms": sum(stages[s]["median_ms"] for s in ("load", "refine", "splat", "classify", "save"))
        }
        results[f"{w}x{h}"] = {"mp": w * h / 1e6, "stages": stages}

    if args.with_depth:
        from nvs.depth.depthpro2 import DepthPro2Backend

        be = DepthPro2Backend()
        rgb, _ = synthetic(4.0, dev)
        t0 = time.perf_counter()
        be.load()
        torch.cuda.synchronize()
        cold = (time.perf_counter() - t0) * 1000
        torch.cuda.reset_peak_memory_stats()
        warm = timeit(lambda: be.infer(rgb, 3000.0), args.runs)
        results["depth_h_4mp"] = {
            "load_ms": cold,
            "infer": warm,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        be.unload()

    print(
        "| size | "
        + " | ".join(["load", "refine", "splat", "classify", "coverage", "save", "total warm"])
        + " |"
    )
    print("|---|" + "---|" * 7)
    for size, r in results.items():
        if size.startswith("depth"):
            continue
        s = r["stages"]
        cells = [
            f"{s[k]['median_ms']:.0f}"
            for k in ("load", "refine", "splat", "classify", "coverage", "save", "total_warm")
        ]
        print(f"| {size} ({r['mp']:.1f} MP) | " + " | ".join(cells) + " |")
    print("(median ms; total warm = load + refine + splat + classify + save, depth cached)")
    if "depth_h_4mp" in results:
        d = results["depth_h_4mp"]
        print(
            f"Depth Pro 2 H+ bf16, 4 MP: load {d['load_ms']:.0f} ms, infer {d['infer']['median_ms']:.0f} ms, "
            f"peak {d['peak_vram_gib']:.2f} GiB"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(
        json.dumps({"gpu": torch.cuda.get_device_name(0), "runs": args.runs, "results": results}, indent=1)
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
