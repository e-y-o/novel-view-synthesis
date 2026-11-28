/**
 * Pure geometry of the preview viewport (no DOM, no GL), unit-tested.
 *
 * Coordinates:
 * - **Canvas pixels** `(x, y)`: device pixels of the canvas backing store, origin top-left, y down, continuous
 *   (a pixel `i` covers `[i, i+1)`).
 * - **Image pixels** `(u, v)`: preview-image pixels with pixel *centres* at integer coordinates, so the image
 *   covers `[-0.5, W-0.5] × [-0.5, H-0.5]` and the principal point is `((W-1)/2, (H-1)/2)` (the T08 convention).
 *
 * The photo is shown "contain"-fitted: uniformly scaled by `scale = min(Wc/W, Hc/H)` and centred. A canvas
 * pixel and an image pixel relate by `x = offsetX + (u + 0.5) * scale` (same for y). At the identity pose the
 * virtual camera (source intrinsics, principal point at the image centre) reproduces exactly this mapping.
 */

/** Result of fitting an image into a canvas. All lengths in canvas pixels. */
export interface ContainFit {
  /** Canvas pixels per image pixel. */
  scale: number;
  /** Canvas x of the image's left edge (u = -0.5). */
  offsetX: number;
  /** Canvas y of the image's top edge (v = -0.5). */
  offsetY: number;
  /** Displayed image size in canvas pixels. */
  width: number;
  height: number;
}

/** Letterbox an `imgW × imgH` image into a `canvasW × canvasH` canvas ("contain"). */
export function containFit(
  canvasW: number,
  canvasH: number,
  imgW: number,
  imgH: number,
): ContainFit {
  if (!(canvasW > 0 && canvasH > 0 && imgW > 0 && imgH > 0)) {
    throw new RangeError("canvas and image sizes must be positive");
  }
  const scale = Math.min(canvasW / imgW, canvasH / imgH);
  const width = imgW * scale;
  const height = imgH * scale;
  return {
    scale,
    offsetX: (canvasW - width) / 2,
    offsetY: (canvasH - height) / 2,
    width,
    height,
  };
}

/** Canvas pixel `(x, y)` to image pixel `(u, v)` (may lie outside the image, in the letterbox). */
export function canvasToImage(fit: ContainFit, x: number, y: number): [number, number] {
  return [(x - fit.offsetX) / fit.scale - 0.5, (y - fit.offsetY) / fit.scale - 0.5];
}

/** Image pixel `(u, v)` to canvas pixel `(x, y)`. Inverse of `canvasToImage`. */
export function imageToCanvas(fit: ContainFit, u: number, v: number): [number, number] {
  return [fit.offsetX + (u + 0.5) * fit.scale, fit.offsetY + (v + 0.5) * fit.scale];
}

/** True when the canvas pixel lies inside the displayed image (not in the letterbox bars). */
export function insideImage(fit: ContainFit, x: number, y: number): boolean {
  return (
    x >= fit.offsetX &&
    x < fit.offsetX + fit.width &&
    y >= fit.offsetY &&
    y < fit.offsetY + fit.height
  );
}

/**
 * Scale factors that map a camera-space point to clip space: `clip.xy = k * (Pc.x, -Pc.y)` before the
 * perspective divide, with `k = (2·scale·f/canvasW, 2·scale·f/canvasH)`. `f` is the preview focal length in
 * image pixels. The principal point sits at the canvas centre of the image, so no offset term is needed.
 */
export function clipScale(
  fit: ContainFit,
  f: number,
  canvasW: number,
  canvasH: number,
): [number, number] {
  return [(2 * fit.scale * f) / canvasW, (2 * fit.scale * f) / canvasH];
}

/** Mesh rows for `cols` columns so cells are square: `round(cols · H / W)`, at least 1. */
export function meshRows(cols: number, imgW: number, imgH: number): number {
  return Math.max(1, Math.round((cols * imgH) / imgW));
}

/**
 * Depth-texel index used by the vertex shader for mesh grid coordinate `g` (0..cols):
 * `clamp(floor(g · size / cols), 0, size − 1)`, in integer arithmetic exactly like the GLSL.
 */
export function gridTexel(g: number, cols: number, size: number): number {
  return Math.min(size - 1, Math.max(0, Math.trunc((g * size) / cols)));
}

/**
 * CPU mirror of the vertex shader's cut test: for every mesh quad, whether `max(z)/min(z)` over its four
 * corner depths exceeds `tau`. `depth` holds metres, row-major `imgW × imgH`. Returns `rows × cols` flags.
 * Used by the identity-pose check on the dev page and by tests.
 */
export function quadCutMask(
  depth: Float32Array,
  imgW: number,
  imgH: number,
  cols: number,
  rows: number,
  tau: number,
): Uint8Array {
  const cut = new Uint8Array(cols * rows);
  const at = (gx: number, gy: number): number =>
    depth[gridTexel(gy, rows, imgH) * imgW + gridTexel(gx, cols, imgW)] as number;
  for (let qy = 0; qy < rows; qy += 1) {
    for (let qx = 0; qx < cols; qx += 1) {
      const a = at(qx, qy);
      const b = at(qx + 1, qy);
      const c = at(qx, qy + 1);
      const d = at(qx + 1, qy + 1);
      cut[qy * cols + qx] = Math.max(a, b, c, d) / Math.min(a, b, c, d) > tau ? 1 : 0;
    }
  }
  return cut;
}
