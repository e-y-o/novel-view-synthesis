"""Render reference views of preview assets with the Python splatter, to compare the browser preview against.

    uv run python scripts/preview_reference.py web/fixtures/two_layers web/fixtures/big/room --yaw 8 -8 [--pitch 0]

For every asset folder (``color.jpg``, ``depth.f16``, ``sky.u8``, ``meta.json``, as written by
``scripts/make_fixture_assets.py`` or ``nvs export-preview``) and every pose, writes ``ref_<yaw>_<pitch>.png`` next
to the assets: an RGBA image of the preview size where alpha is 255 on pixels the splatter filled and 0 on holes.
The pose is ``orbit_pose(yaw, pitch, pivot_z)`` and the camera uses the preview's own size and ``f_px_preview``,
exactly what the viewer does. The dev page (``web/dev/renderer.html``) loads these files in
``__dev.compareWithReference`` and reports the mean absolute error on the pixels both renderers fill.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def render_reference(folder: Path, yaw: float, pitch: float, device: str | None = None) -> Path:
    import numpy as np
    import torch
    from PIL import Image

    from nvs.core.camera import make_camera, orbit_pose
    from nvs.render.splat import SplatOptions, splat

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((folder / "meta.json").read_text())
    w, h = meta["preview_width"], meta["preview_height"]
    rgb = np.asarray(Image.open(folder / "color.jpg").convert("RGB"), dtype=np.float32) / 255
    depth = np.frombuffer((folder / "depth.f16").read_bytes(), dtype="<f2").reshape(h, w).astype(np.float32)
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(device)
    depth_t = torch.from_numpy(depth).to(device)
    R, t = orbit_pose(yaw, pitch, meta["pivot_z"])
    cam = make_camera(R, t, meta["f_px_preview"], w, h)
    res = splat(rgb_t, depth_t, cam, options=SplatOptions(extra_channels=False))
    image = (res.image.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    alpha = ((~res.hole).byte() * 255).cpu().numpy()[:, :, None]
    out = folder / f"ref_{yaw:g}_{pitch:g}.png"
    Image.fromarray(np.concatenate([image, alpha], axis=2), "RGBA").save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--yaw", nargs="+", type=float, default=[8.0, -8.0])
    p.add_argument("--pitch", type=float, default=0.0)
    args = p.parse_args(argv)
    for folder in args.folders:
        for yaw in args.yaw:
            print(render_reference(folder, yaw, args.pitch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
