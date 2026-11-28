"""Synthetic ground-truth scenes: a tiny analytic ray-caster in torch (test helper, not installed).

Everything uses the shared camera convention (nvs.core.camera): OpenCV axes (x right, y down, z forward),
source camera = identity at the origin, target pose P_target = R^T (P - t). Depth is **z-depth** (along the
camera axis), not ray length: rays are built with z = 1, so the ray parameter *is* the z-depth.

Scenes (units: metres; the camera sits at the origin, y is down)::

    scene_two_layers()                 scene_room()                  scene_sky_tree()
    top view (x right, z up)           side view (y down, z right)   top view
      wall  z = 4  ===========           ceiling: none                 sky      (rays that miss = inf depth)
      box   z 1.5..2.5 [##]              wall  z = 6  |                poles   ||   |   ||  |   (thin boxes,
      camera  ^ (0,0,0)                  sphere r .6  O  (rests on     z 2..6, 8 cm wide, y -5..1.5)
                                         floor y = 1.2 ______/ floor   ground  y = 1.5 plane

Textures are procedural and view-independent (functions of the 3D hit point): 3D checkerboard, smooth
gradient and a few seeded Gaussian colour blobs, so matching errors show up. All randomness is seeded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from nvs.core.types import CameraSpec

_EPS = 1e-4


@dataclass
class Texture:
    tone_a: tuple[float, float, float]
    tone_b: tuple[float, float, float]
    cell: float = 0.25
    seed: int = 0
    n_blobs: int = 40
    blob_radius: float = 0.25
    shift: torch.Tensor = field(init=False)
    blob_centres: torch.Tensor = field(init=False)
    blob_colours: torch.Tensor = field(init=False)
    grad_dir: torch.Tensor = field(init=False)

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        self.shift = torch.rand(3, generator=g, dtype=torch.float64) * self.cell + 0.0137
        lo = torch.tensor([-3.0, -1.5, 0.8], dtype=torch.float64)
        hi = torch.tensor([3.0, 1.5, 6.0], dtype=torch.float64)
        self.blob_centres = lo + (hi - lo) * torch.rand(self.n_blobs, 3, generator=g, dtype=torch.float64)
        self.blob_colours = torch.rand(self.n_blobs, 3, generator=g, dtype=torch.float64)
        self.grad_dir = torch.randn(3, generator=g, dtype=torch.float64)

    def colour(self, P: torch.Tensor) -> torch.Tensor:
        """World points P[N,3] -> RGB[N,3] in [0,1]."""
        dt, dev = P.dtype, P.device
        a, b = (torch.tensor(c, dtype=dt, device=dev) for c in (self.tone_a, self.tone_b))
        checker = (torch.floor((P + self.shift.to(dt).to(dev)) / self.cell).sum(-1) % 2).unsqueeze(-1)
        base = a * (1 - checker) + b * checker
        base = base * (0.8 + 0.2 * torch.sin(P @ self.grad_dir.to(dt).to(dev)).unsqueeze(-1))
        d2 = torch.cdist(P, self.blob_centres.to(dt).to(dev)) ** 2
        w = torch.exp(-d2 / (2 * self.blob_radius**2))
        total = w.sum(-1, keepdim=True)
        cover = total.clamp(max=1.0)
        blob = (w @ self.blob_colours.to(dt).to(dev)) / (total + 1e-6)
        return (base * (1 - cover) + blob * cover).clamp(0, 1)


@dataclass
class Plane:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    texture: Texture
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None  # world AABB limit

    def intersect(self, o, d):
        n = torch.tensor(self.normal, dtype=d.dtype, device=d.device)
        p0 = torch.tensor(self.point, dtype=d.dtype, device=d.device)
        denom = d @ n
        safe = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
        t = ((p0 - o) @ n) / safe
        ok = (denom.abs() >= 1e-9) & (t > _EPS)
        if self.bounds is not None:
            P = o + t.unsqueeze(-1) * d
            lo, hi = (torch.tensor(b, dtype=d.dtype, device=d.device) for b in self.bounds)
            ok &= ((P >= lo) & (P <= hi)).all(-1)
        return torch.where(ok, t, torch.full_like(t, float("inf")))


@dataclass
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    texture: Texture

    def intersect(self, o, d):
        lo, hi = (torch.tensor(b, dtype=d.dtype, device=d.device) for b in (self.lo, self.hi))
        d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t1, t2 = (lo - o) / d, (hi - o) / d
        tmin = torch.minimum(t1, t2).amax(-1)
        tmax = torch.maximum(t1, t2).amin(-1)
        t = torch.where(tmin > _EPS, tmin, tmax)
        ok = (tmax >= tmin) & (tmax > _EPS)
        return torch.where(ok, t, torch.full_like(t, float("inf")))


@dataclass
class Sphere:
    centre: tuple[float, float, float]
    radius: float
    texture: Texture

    def intersect(self, o, d):
        c = torch.tensor(self.centre, dtype=d.dtype, device=d.device)
        oc = o - c
        a = (d * d).sum(-1)
        hb = d @ oc
        disc = hb * hb - a * ((oc * oc).sum() - self.radius**2)
        sq = torch.sqrt(disc.clamp_min(0))
        t0, t1 = (-hb - sq) / a, (-hb + sq) / a
        t = torch.where(t0 > _EPS, t0, t1)
        ok = (disc >= 0) & (t > _EPS)
        return torch.where(ok, t, torch.full_like(t, float("inf")))


@dataclass
class Scene:
    primitives: list
    sky: bool = False


def _sky_colour(dir_world: torch.Tensor) -> torch.Tensor:
    n = dir_world / dir_world.norm(dim=-1, keepdim=True)
    s = (-n[:, 1]).clamp(0, 1).unsqueeze(-1)  # y is down: up = negative y
    horizon = torch.tensor([0.80, 0.88, 0.95], dtype=n.dtype, device=n.device)
    zenith = torch.tensor([0.25, 0.45, 0.85], dtype=n.dtype, device=n.device)
    return horizon * (1 - s) + zenith * s


def _rays(cam: CameraSpec, dtype, device):
    W, H = cam.width, cam.height
    v, u = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing="ij",
    )
    x = (u - (W - 1) / 2) / cam.f_px
    y = (v - (H - 1) / 2) / cam.f_px
    d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)  # z = 1: ray parameter = z-depth
    R = cam.R.to(dtype=dtype, device=device)
    return cam.t.to(dtype=dtype, device=device), d_cam @ R.T, d_cam


def render(scene: Scene, cam: CameraSpec, W: int | None = None, H: int | None = None, device="cpu"):
    """Ray-cast ``scene`` from ``cam``. Returns (color[3,H,W] float32, depth[H,W] float32 z-depth with inf
    for sky/miss, hit_id[H,W] int64: primitive index, -1 for sky/miss). W, H must match the camera."""
    if (W is not None and W != cam.width) or (H is not None and H != cam.height):
        raise ValueError("W/H must match the camera size")
    W, H = cam.width, cam.height
    dtype = torch.float64
    o, d, _ = _rays(cam, dtype, device)
    ts = torch.stack([p.intersect(o, d) for p in scene.primitives], dim=0)  # [K, N]
    depth, idx = ts.min(dim=0)
    hit = torch.isfinite(depth)
    idx = torch.where(hit, idx, torch.full_like(idx, -1))
    colour = torch.zeros(d.shape[0], 3, dtype=dtype, device=device)
    P = o + depth.unsqueeze(-1).where(hit.unsqueeze(-1), torch.zeros_like(depth).unsqueeze(-1)) * d
    for k, prim in enumerate(scene.primitives):
        m = idx == k
        if m.any():
            colour[m] = prim.texture.colour(P[m])
    sky = ~hit
    if scene.sky and sky.any():
        colour[sky] = _sky_colour(d[sky])
    return (
        colour.T.reshape(3, H, W).float(),
        depth.reshape(H, W).float(),
        idx.reshape(H, W),
    )


def visible_in_both(scene: Scene, cam_src: CameraSpec, cam_tgt: CameraSpec, W=None, H=None, device="cpu"):
    """bool[H,W] over target pixels: True where the target surface point is also visible from the source
    (projects inside the source frame and its z matches the source depth within 1 %). Sky pixels count as
    visible when the source ray in the same direction is also sky. False = legitimate hole."""
    _, depth_s, _ = render(scene, cam_src, device=device)
    _, depth_t, _ = render(scene, cam_tgt, device=device)
    dtype = torch.float64
    o, d, d_cam = _rays(cam_tgt, dtype, device)
    z = depth_t.reshape(-1).to(dtype)
    tgt_sky = ~torch.isfinite(z)
    z_eff = torch.where(tgt_sky, torch.full_like(z, 1e6), z)  # sky: a far point along the ray
    P = o + z_eff.unsqueeze(-1) * d
    Rs = cam_src.R.to(dtype=dtype, device=device)
    Q = (P - cam_src.t.to(dtype=dtype, device=device)) @ Rs  # R_s^T (P - t_s)
    Ws, Hs, f = cam_src.width, cam_src.height, cam_src.f_px
    Qz = Q[:, 2]
    in_front = Qz > _EPS
    safe = torch.where(in_front, Qz, torch.ones_like(Qz))
    u = f * Q[:, 0] / safe + (Ws - 1) / 2
    v = f * Q[:, 1] / safe + (Hs - 1) / 2
    ui, vi = torch.round(u).long(), torch.round(v).long()
    inside = in_front & (ui >= 0) & (ui < Ws) & (vi >= 0) & (vi < Hs)
    zs = depth_s.reshape(-1).to(dtype)[(vi.clamp(0, Hs - 1) * Ws + ui.clamp(0, Ws - 1))]
    src_sky = ~torch.isfinite(zs)
    match = torch.where(
        tgt_sky | src_sky,
        tgt_sky & src_sky,
        (Qz - zs).abs() <= 0.01 * torch.where(src_sky, torch.ones_like(zs), zs),
    )
    return (inside & match).reshape(cam_tgt.height, cam_tgt.width)


def psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """PSNR [dB] between images a, b [C,H,W] in [0,1] over mask[H,W] bool (all pixels if None)."""
    err = (a - b) ** 2
    if mask is not None:
        err = err[:, mask]
    mse = float(err.mean())
    return math.inf if mse == 0 else -10 * math.log10(mse)


def scene_two_layers() -> Scene:
    """Back wall at z = 4 m and a 1 m wide box in front (front face at z = 1.5 m)."""
    wall = Plane((0, 0, 4.0), (0, 0, -1.0), Texture((0.85, 0.8, 0.7), (0.5, 0.45, 0.4), seed=1))
    box = Box((-0.5, -0.5, 1.5), (0.5, 0.5, 2.5), Texture((0.9, 0.2, 0.2), (0.3, 0.1, 0.1), cell=0.2, seed=2))
    return Scene([wall, box])


def scene_room() -> Scene:
    """Floor (y = 1.2, grazing), back wall at z = 6 m and a sphere resting on the floor."""
    floor = Plane((0, 1.2, 0), (0, -1.0, 0), Texture((0.7, 0.7, 0.75), (0.3, 0.3, 0.35), cell=0.5, seed=3))
    wall = Plane((0, 0, 6.0), (0, 0, -1.0), Texture((0.9, 0.85, 0.6), (0.6, 0.55, 0.3), seed=4))
    sphere = Sphere((0.3, 0.6, 2.5), 0.6, Texture((0.2, 0.5, 0.9), (0.1, 0.2, 0.5), cell=0.15, seed=5))
    return Scene([floor, wall, sphere])


def scene_sky_tree() -> Scene:
    """Thin vertical poles at various depths against sky, over a ground plane."""
    tex = Texture((0.35, 0.25, 0.15), (0.2, 0.15, 0.1), cell=0.2, seed=6)
    ground = Plane((0, 1.5, 0), (0, -1.0, 0), Texture((0.3, 0.5, 0.25), (0.2, 0.35, 0.15), cell=0.5, seed=7))
    poles = [
        Box((x - 0.04, -5.0, z), (x + 0.04, 1.5, z + 0.08), tex)
        for x, z in [(-1.2, 2.0), (-0.3, 3.5), (0.4, 2.5), (1.1, 5.0), (1.8, 6.0)]
    ]
    return Scene([ground, *poles], sky=True)
