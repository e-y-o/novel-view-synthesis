"""End-to-end ground-truth test: the CLI code path (``run_render`` + ``--depth-file``) on analytic scenes.

For each T03 scene and orbit the source photo and its exact depth are written to disk, ``run_render`` produces
the new view, and it is compared with the ray-cast ground truth on the pixels that are visible from both
cameras (and not marked as holes). Holes must also cover the true disocclusions away from their edges.

Measured on this machine (512x384, f = 500 px; PSNR dB on visible & non-hole pixels, then the share of the
true-disocclusion interior (>= 3 px from visible pixels) that is marked as hole). Distant textured surfaces
(> 12 m) are excluded and the main PSNR skips 2 px around silhouettes (on the sky/poles scene the silhouette
pixels of the 8 cm poles are 8 % of the pixels and cost ~12 dB; away from them it is ~40 dB), see
``run_case``. Run ``pytest -s tests/test_e2e_synthetic.py`` to print the table; the asserted floors are
28 dB and 90 %.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from synth import psnr, render, scene_room, scene_sky_tree, scene_two_layers, visible_in_both

from nvs.cli.main import build_parser
from nvs.cli.render import run_render
from nvs.core.camera import make_camera, orbit_pose

W, H, F_PX = 512, 384, 500.0
SRC = make_camera(torch.eye(3), torch.zeros(3), F_PX, W, H)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MARGIN = 3  # px: true-disocclusion pixels closer than this to a visible pixel are ignored
MIN_PSNR = 28.0  # away from silhouettes
MIN_PSNR_ALL = 24.0  # sanity floor including silhouette pixels (thin poles are mostly edge)
EDGE_PX = 2
FAR_M = 12.0
SCENES = {
    "two_layers": (scene_two_layers, 2.5),
    "room": (scene_room, 2.5),
    "sky_tree": (scene_sky_tree, 3.5),
}
POSES = [(-8, 0), (-3, 0), (3, 0), (8, 0), (4, 5)]


def run_case(tmp_path, name, yaw, pitch):
    make_scene, pivot = SCENES[name]
    scene = make_scene()
    color, depth, _ = render(scene, SRC)
    img = tmp_path / "src.png"
    Image.fromarray((color.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(img)
    np.save(tmp_path / "depth.npy", depth.numpy())  # sky is inf: the CLI clamps it like a real depth map
    out = tmp_path / "out.png"
    args = build_parser().parse_args(
        ["render", str(img), "--depth-file", str(tmp_path / "depth.npy"), "--f-px", str(F_PX),
         "--device", DEVICE,
         "--yaw", str(yaw), "--pitch", str(pitch), "--pivot", str(pivot), "--out", str(out)]
    )  # fmt: skip
    report = run_render(args)
    result = np.asarray(Image.open(out))
    got = torch.from_numpy(result[:, :, :3]).permute(2, 0, 1).float() / 255
    opaque = torch.from_numpy(result[:, :, 3] > 0)

    R, t = orbit_pose(yaw, pitch, pivot)
    cam = make_camera(R, t, F_PX, W, H)
    gt, gt_depth, gt_hit = render(scene, cam)
    edge = torch.zeros_like(gt_hit, dtype=torch.bool)  # silhouettes of the ground-truth view
    edge[:, 1:] |= gt_hit[:, 1:] != gt_hit[:, :-1]
    edge[1:, :] |= gt_hit[1:, :] != gt_hit[:-1, :]
    interior = ~(F.max_pool2d(edge[None, None].float(), 2 * EDGE_PX + 1, 1, EDGE_PX)[0, 0] > 0)
    # Distant textured surfaces (finite depth > 12 m, i.e. the ground near the horizon) are left out: their
    # checker cells are sub-pixel (aliasing dominates PSNR) and the 1 % depth test of visible_in_both is
    # below one pixel of depth resolution there, so it misreports them as hidden. Sky (inf) stays in.
    near = ~(torch.isfinite(gt_depth) & (gt_depth > FAR_M))
    vis = visible_in_both(scene, SRC, cam)
    scored = vis & opaque & near
    quality = psnr(got, gt, scored & interior)
    quality_all = psnr(got, gt, scored)
    far = (
        ~(F.max_pool2d(vis[None, None].float(), 2 * MARGIN + 1, 1, MARGIN)[0, 0] > 0) & near
    )  # >= MARGIN px away
    hole_share = float((~opaque & far).sum() / max(1, int(far.sum()))) if far.any() else 1.0
    return report, quality, quality_all, hole_share


@pytest.mark.parametrize("name", list(SCENES))
@pytest.mark.parametrize("yaw,pitch", POSES)
def test_end_to_end_against_ground_truth(tmp_path, name, yaw, pitch):
    report, quality, quality_all, hole_share = run_case(tmp_path, name, yaw, pitch)
    print(
        f"{name:11s} yaw {yaw:+d} pitch {pitch:+d}: PSNR interior {quality:.1f} dB, "
        f"all {quality_all:.1f} dB, "
        f"hole share {hole_share:.3f}"
    )
    assert (report.width, report.height) == (W, H)
    assert quality >= MIN_PSNR
    assert quality_all >= MIN_PSNR_ALL
    assert hole_share >= 0.90
