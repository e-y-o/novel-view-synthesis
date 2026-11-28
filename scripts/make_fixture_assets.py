"""Write synthetic preview assets in the exact M2 contract for frontend work (no model, no server needed).

    uv run python scripts/make_fixture_assets.py [--scenes two_layers room sky_tree] [--size 1024 768]
                                                  [--out web/fixtures]    (or: cd web && npm run fixtures)

Each scene of ``tests/synth.py`` is ray-cast at ``--size`` and packaged with ``nvs.preview.assets.package_preview``,
so ``web/fixtures/<scene>/`` holds ``color.jpg``, ``depth.f16``, ``sky.u8`` and ``meta.json`` exactly as the server
would write them. ``sky_tree`` has thin poles against the sky. The output folder is git-ignored; in dev the files
are served by Vite at ``/fixtures/<scene>/...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("two_layers", "room", "sky_tree")


def make_fixture(scene_name: str, out_dir: Path, width: int = 1024, height: int = 768) -> dict:
    """Render one synthetic scene and write its preview assets to ``out_dir``; returns the meta dict."""
    sys.path.insert(0, str(ROOT / "tests"))
    import synth
    import torch

    from nvs.core.camera import make_camera
    from nvs.core.types import DepthResult, Intrinsics
    from nvs.depth.refine import sky_and_clamp
    from nvs.preview.assets import package_preview

    scene = getattr(synth, f"scene_{scene_name}")()
    f_px = 0.9 * max(width, height)
    cam = make_camera(torch.eye(3), torch.zeros(3), f_px, width, height)
    color, depth, _ = synth.render(scene, cam, width, height)
    depth, sky, notes = sky_and_clamp(depth)  # sky rays are inf: same clamp the real pipeline applies
    result = DepthResult(depth, f_px, sky, "synthetic", (height, width), notes)
    intrinsics = Intrinsics(f_px, "override", False, [])
    return package_preview(
        color,
        result,
        intrinsics,
        out_dir,
        filename=f"{scene_name}.synthetic",
        warnings=["synthetic scene (tests/synth.py)", *notes],
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), default=(1024, 768))
    p.add_argument("--out", type=Path, default=ROOT / "web" / "fixtures")
    args = p.parse_args(argv)
    for name in args.scenes:
        info = make_fixture(name, args.out / name, *args.size)
        print(
            f"{name}: {info['preview_width']}x{info['preview_height']}, f={info['f_px_source']:.0f} px, "
            f"pivot {info['pivot_z']:.2f} m -> {args.out / name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
