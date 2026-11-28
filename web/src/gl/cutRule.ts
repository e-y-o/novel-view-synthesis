/**
 * CPU reference of the vertex shader's per-quad cut rule (`shaders/preview.vert.glsl`), in float64.
 *
 * It exists to test the logic on small arrays and to measure cut statistics (for example how many quads are cut at
 * the identity pose) without a GPU. Keep it in step with the shader.
 *
 * The rule ("stretch" mode): for each of a quad's four edges, the projected length in the target view (preview
 * pixels) is divided by what a surface continuing through the quad would cover, `cell · (z / zt)` with `cell` the
 * mesh cell size in preview pixels and `z`, `zt` the depth of the endpoints in the source and in the target view
 * (their mean for the edge). A smooth surface, however steep, stays near 1; a quad spanning a depth jump
 * stretches as soon as the camera moves; at the identity pose every quad is exactly 1. The quad's alpha is
 * `1 − smoothstep(fade, cut, stretch)` (a hard step at `cut` without feathering); both thresholds are multiplied
 * by `stretchSmoothFactor` on quads without a depth jump (depth ratio under about 1.04, blended up to 1.2). Quads entirely in the sky are
 * never removed; with `skyBoundaryCut` quads that straddle sky and non-sky always are.
 */
import { transformToTarget, type CameraSpec } from "../camera";
import { gridTexel } from "./layout";
import type { DrawOptions } from "./PreviewRenderer";

/** The cut-related subset of `DrawOptions`. */
export type CutParams = Pick<
  DrawOptions,
  | "cutMode"
  | "cutTau"
  | "stretchFade"
  | "stretchCut"
  | "stretchSmoothFactor"
  | "feather"
  | "skyBoundaryCut"
>;

export const DEFAULT_CUT_PARAMS: CutParams = {
  cutMode: "stretch",
  cutTau: 1.08,
  stretchFade: 2,
  stretchCut: 5,
  stretchSmoothFactor: 4,
  feather: true,
  skyBoundaryCut: false,
};

const NEAR = 0.02;
const BEHIND = 1e6;

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** Geometry shared by every quad of a mesh. */
export interface MeshGeometry {
  /** Preview size in pixels. */
  width: number;
  height: number;
  cols: number;
  rows: number;
  /** Preview focal length in pixels. */
  f: number;
}

export interface QuadResult {
  /** 1 = fully drawn, 0 = removed, in between = feathered. */
  alpha: number;
  /** Largest edge stretch (1 = unstretched); `NaN` in "tau" mode and for sky-only / sky-boundary quads. */
  stretch: number;
}

/** Stretch of one edge: projected length over `cell · mean(z/zt)`; `1e6` if an endpoint is behind the camera. */
export function edgeStretch(
  a: [number, number, number],
  b: [number, number, number],
  za: number,
  zb: number,
  cell: number,
  f: number,
): number {
  if (a[2] < NEAR || b[2] < NEAR) return BEHIND;
  const len = Math.hypot(f * (a[0] / a[2] - b[0] / b[2]), f * (a[1] / a[2] - b[1] / b[2]));
  return len / (cell * 0.5 * (za / a[2] + zb / b[2]));
}

/** Alpha of quad `(qx, qy)` of the mesh, like the vertex shader computes it. */
export function quadAlpha(
  depth: Float32Array,
  sky: Uint8Array,
  geom: MeshGeometry,
  qx: number,
  qy: number,
  pose: CameraSpec,
  params: CutParams = DEFAULT_CUT_PARAMS,
): QuadResult {
  const { width: W, height: H, cols, rows, f } = geom;
  const corners: [number, number][] = [
    [qx, qy],
    [qx + 1, qy],
    [qx, qy + 1],
    [qx + 1, qy + 1],
  ];
  const texel = corners.map(([gx, gy]) => gridTexel(gy, rows, H) * W + gridTexel(gx, cols, W));
  const z = texel.map((i) => depth[i] as number);
  if (params.cutMode === "tau") {
    const ratio = Math.max(...z) / Math.min(...z);
    return { alpha: ratio > params.cutTau ? 0 : 1, stretch: NaN };
  }
  const isSky = texel.map((i) => (sky[i] as number) > 127);
  if (isSky.every(Boolean)) return { alpha: 1, stretch: NaN };
  if (params.skyBoundaryCut && isSky.some(Boolean)) return { alpha: 0, stretch: NaN };

  const cell = W / cols;
  const cx = (W - 1) / 2;
  const cy = (H - 1) / 2;
  const cam = corners.map(([gx, gy], i) => {
    const px = (gx * W) / cols - 0.5;
    const py = (gy * H) / rows - 0.5;
    const zi = z[i] as number;
    return transformToTarget([((px - cx) / f) * zi, ((py - cy) / f) * zi, zi], pose.R, pose.t);
  });
  const [p00, p10, p01, p11] = cam as [
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  const [z00, z10, z01, z11] = z as [number, number, number, number];
  const stretch = Math.max(
    edgeStretch(p00, p10, z00, z10, cell, f),
    edgeStretch(p01, p11, z01, z11, cell, f),
    edgeStretch(p00, p01, z00, z01, cell, f),
    edgeStretch(p10, p11, z10, z11, cell, f),
  );
  // no depth jump: the surface may stretch more before it is dropped (see the shader)
  const k =
    Math.max(1, params.stretchSmoothFactor) +
    (1 - Math.max(1, params.stretchSmoothFactor)) *
      smoothstep(1.04, 1.2, Math.max(...z) / Math.min(...z));
  const fadeAt = params.stretchFade * k;
  const cutAt = Math.max(params.stretchCut, params.stretchFade + 1e-3) * k;
  const alpha = params.feather ? 1 - smoothstep(fadeAt, cutAt, stretch) : stretch > cutAt ? 0 : 1;
  return { alpha, stretch };
}

export interface CutStats {
  total: number;
  /** Quads with alpha ≈ 0 (the shader discards them). */
  removed: number;
  /** Quads drawn translucent (0 < alpha < 1). */
  feathered: number;
  removedFraction: number;
  featheredFraction: number;
}

/** Count removed and feathered quads over the whole mesh (optionally every `step`-th quad per axis). */
export function cutStats(
  depth: Float32Array,
  sky: Uint8Array,
  geom: MeshGeometry,
  pose: CameraSpec,
  params: CutParams = DEFAULT_CUT_PARAMS,
  step = 1,
): CutStats {
  let total = 0;
  let removed = 0;
  let feathered = 0;
  for (let qy = 0; qy < geom.rows; qy += step) {
    for (let qx = 0; qx < geom.cols; qx += step) {
      const { alpha } = quadAlpha(depth, sky, geom, qx, qy, pose, params);
      total += 1;
      if (alpha <= 0.002) removed += 1;
      else if (alpha < 0.998) feathered += 1;
    }
  }
  return {
    total,
    removed,
    feathered,
    removedFraction: removed / total,
    featheredFraction: feathered / total,
  };
}
