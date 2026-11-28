"""Synthetic correctness test for nvs.core.splat (plane at 3 m, square at 1 m, camera 0.1 m right)."""

import torch

from nvs.core import splat as mod

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F, W, H = 500.0, 512, 384


def centroid_x(weight):
    cols = torch.arange(weight.shape[1], dtype=torch.float32, device=weight.device)
    return float((weight.sum(0) * cols).sum() / weight.sum())


def build():
    color = torch.full((3, H, W), 0.5, device=DEV)
    depth = torch.full((H, W), 3.0, device=DEV)
    color[:, :, 100:140] = torch.tensor([1.0, 0, 0], device=DEV)[:, None, None]  # red stripe on the plane
    color[:, 140:260, 200:300] = torch.tensor([0, 0, 1.0], device=DEV)[:, None, None]  # blue square
    depth[140:260, 200:300] = 1.0
    return color, depth


def test_shifts_and_disocclusion():
    color, depth = build()
    img, hole = mod.splat(color, depth, F, torch.eye(3), torch.tensor([0.1, 0.0, 0.0]))
    red = ((img[0] > 0.8) & (img[1] < 0.2) & (img[2] < 0.2)).float()
    blue = ((img[2] > 0.8) & (img[0] < 0.2)).float()
    assert abs(centroid_x(red) - (119.5 - F * 0.1 / 3)) <= 0.5  # plane shifts left by f*0.1/3 = 16.67 px
    assert abs(centroid_x(blue) - (249.5 - F * 0.1 / 1)) <= 0.5  # square shifts left by f*0.1/1 = 50 px
    row = slice(180, 220)
    new_right = 299.5 - F * 0.1  # square's new right edge
    right_strip = hole[
        row, int(new_right) + 6 : int(new_right) + 28
    ]  # bg revealed on the right of the square
    left_strip = hole[row, int(new_right) - 130 : int(new_right) - 110]
    assert right_strip.float().mean() > 0.95
    assert left_strip.float().mean() < 0.01
