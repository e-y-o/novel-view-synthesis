import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  defaultOrbit,
  identityPose,
  orbitPose,
  decodeHalfArray,
  type CameraSpec,
} from "../src/camera";
import {
  DEFAULT_CUT_PARAMS,
  cutStats,
  edgeStretch,
  quadAlpha,
  smoothstep,
  type CutParams,
  type MeshGeometry,
} from "../src/gl/cutRule";

const N = 16;
const geom: MeshGeometry = { width: N, height: N, cols: N, rows: N, f: 100 };
const noSky = new Uint8Array(N * N);
const TAU: CutParams = { ...DEFAULT_CUT_PARAMS, cutMode: "tau" };
const STRETCH = DEFAULT_CUT_PARAMS;

const make = (fn: (x: number, y: number) => number): Float32Array =>
  Float32Array.from({ length: N * N }, (_, i) => fn(i % N, Math.floor(i / N)));
const yaw = (deg: number, pivot = 3): CameraSpec =>
  orbitPose({ ...defaultOrbit(pivot), yawDeg: deg });

/** Alpha of every quad, as rows of columns (quads are N x N: grid coordinate g spans 0..N, quad g..g+1). */
function alphas(depth: Float32Array, pose: CameraSpec, params: CutParams, sky = noSky): number[][] {
  return Array.from({ length: N }, (_, qy) =>
    Array.from({ length: N }, (_, qx) => quadAlpha(depth, sky, geom, qx, qy, pose, params).alpha),
  );
}
const countRemoved = (a: number[][]): number => a.flat().filter((v) => v <= 0.002).length;

describe("smoothstep", () => {
  it("is 0 below, 1 above and smooth between", () => {
    expect(smoothstep(1, 3, 0)).toBe(0);
    expect(smoothstep(1, 3, 5)).toBe(1);
    expect(smoothstep(1, 3, 2)).toBe(0.5);
    expect(smoothstep(1, 3, 1.5)).toBeLessThan(0.5);
  });
});

describe("edgeStretch", () => {
  it("is 1 for an unmoved edge and grows with parallax", () => {
    const a: [number, number, number] = [0, 0, 4];
    const b: [number, number, number] = [0.04, 0, 4]; // one pixel apart at f = 100
    expect(edgeStretch(a, b, 4, 4, 1, 100)).toBeCloseTo(1, 12);
    expect(edgeStretch(a, [0.08, 0, 4], 4, 4, 1, 100)).toBeCloseTo(2, 12);
  });
  it("accounts for the change of scale (dolly) through z / zt", () => {
    // everything moves twice as close: lengths double, expected doubles too
    const a: [number, number, number] = [0, 0, 2];
    const b: [number, number, number] = [0.04, 0, 2];
    expect(edgeStretch(a, b, 4, 4, 1, 100)).toBeCloseTo(1, 12);
  });
  it("flags points behind the camera", () => {
    expect(edgeStretch([0, 0, -1], [0, 0, 2], 3, 3, 1, 100)).toBe(1e6);
  });
});

describe("identity pose", () => {
  it("removes nothing in stretch mode, however jagged the depth", () => {
    // checkerboard of 2 m and 20 m: the worst case for a ratio test
    const depth = make((x, y) => ((x + y) % 2 === 0 ? 2 : 20));
    const a = alphas(depth, identityPose(), STRETCH);
    expect(a.flat().every((v) => Math.abs(v - 1) < 1e-9)).toBe(true);
    // the legacy rule cuts all of it (the corner quad clamped at the border is uniform)
    expect(countRemoved(alphas(depth, identityPose(), TAU))).toBeGreaterThanOrEqual(N * N - 1);
  });
});

describe("step edge", () => {
  const depth = make((x) => (x < 8 ? 2 : 6)); // columns 0-7 near, 8-15 far: the step lies in quad column 7

  it("stays connected at the identity pose and is removed once the camera moves", () => {
    expect(countRemoved(alphas(depth, identityPose(), STRETCH))).toBe(0);
    const moved = alphas(depth, yaw(8), STRETCH);
    for (let qy = 0; qy < N; qy += 1) {
      expect(moved[qy]![7]).toBeLessThanOrEqual(0.002); // the quad column spanning the step
      expect(moved[qy]![3]).toBeGreaterThan(0.998);
      expect(moved[qy]![12]).toBeGreaterThan(0.998);
    }
  });

  it("fades smoothly: alpha of the step quad never rises as the camera moves, and passes through (0, 1)", () => {
    let prev = 1;
    let sawFeather = false;
    for (let d = 0; d <= 12; d += 0.05) {
      const a = quadAlpha(depth, noSky, geom, 7, 5, yaw(d), STRETCH).alpha;
      expect(a).toBeLessThanOrEqual(prev + 1e-9);
      if (a > 0.01 && a < 0.99) sawFeather = true;
      prev = a;
    }
    expect(sawFeather).toBe(true);
    expect(prev).toBeLessThanOrEqual(0.002);
  });

  it("without feathering the alpha is 0 or 1 only", () => {
    const hard = { ...STRETCH, feather: false };
    for (let d = 0; d <= 12; d += 0.37) {
      const a = quadAlpha(depth, noSky, geom, 7, 5, yaw(d), hard).alpha;
      expect(a === 0 || a === 1).toBe(true);
    }
  });
});

describe("slope", () => {
  // A plane: inverse depth is linear in the image. Steep: depth goes from 2 m to 8 m across the quad row.
  const depth = make((x) => 1 / (0.5 - (0.375 * x) / (N - 1)));

  it("is kept by the stretch rule but torn by the global ratio test", () => {
    const ratio = Math.max(...depth) / Math.min(...depth);
    expect(ratio).toBeGreaterThan(3);
    // the steeper half of the ramp exceeds the global ratio (1.08) between neighbours
    expect(countRemoved(alphas(depth, identityPose(), TAU))).toBeGreaterThan(N * N * 0.4);
    expect(countRemoved(alphas(depth, identityPose(), STRETCH))).toBe(0);
    // a small move: the connected surface does not tear
    const moved = alphas(depth, yaw(3), STRETCH);
    expect(countRemoved(moved)).toBe(0);
    expect(Math.min(...moved.flat())).toBeGreaterThan(0.5);
  });
});

describe("sky", () => {
  const sky = new Uint8Array(N * N);
  const depth = make((x) => (x < 8 ? 3 : 50));
  for (let y = 0; y < N; y += 1) for (let x = 8; x < N; x += 1) sky[y * N + x] = 255;

  it("never removes quads fully inside the sky, even over wild depth", () => {
    const wild = make((x, y) => (x < 8 ? 3 : (x + y) % 2 === 0 ? 20 : 200));
    const a = alphas(wild, yaw(10), STRETCH, sky);
    for (let qy = 0; qy < N; qy += 1)
      for (let qx = 8; qx < N - 1; qx += 1) expect(a[qy]![qx]).toBe(1);
  });

  it("keeps the sky boundary connected at identity and removes it when moving (stretch rule)", () => {
    expect(countRemoved(alphas(depth, identityPose(), STRETCH, sky))).toBe(0);
    const moved = alphas(depth, yaw(8), STRETCH, sky);
    for (let qy = 0; qy < N; qy += 1) expect(moved[qy]![7]).toBeLessThanOrEqual(0.002);
  });

  it("skyBoundaryCut removes exactly the straddling quads, even at identity", () => {
    const a = alphas(depth, identityPose(), { ...STRETCH, skyBoundaryCut: true }, sky);
    for (let qy = 0; qy < N; qy += 1) {
      for (let qx = 0; qx < N; qx += 1) expect(a[qy]![qx]).toBe(qx === 7 ? 0 : 1);
    }
  });
});

describe("single outlier corner", () => {
  const depth = make((x, y) => (x === 8 && y === 8 ? 1 : 4)); // one wrong pixel in a flat wall

  it("is invisible at identity and isolated when moving: only the four quads touching it go", () => {
    expect(countRemoved(alphas(depth, identityPose(), STRETCH))).toBe(0);
    const moved = alphas(depth, yaw(6), STRETCH);
    const gone: string[] = [];
    moved.forEach((row, qy) => row.forEach((v, qx) => v <= 0.002 && gone.push(`${qx},${qy}`)));
    expect(gone.sort()).toEqual(["7,7", "7,8", "8,7", "8,8"]);
  });
});

describe("behind the camera", () => {
  it("removes quads that end up behind the target camera", () => {
    const depth = make(() => 4);
    const pose: CameraSpec = { R: identityPose().R, t: [0, 0, 10] }; // 6 m past the surface
    expect(countRemoved(alphas(depth, pose, STRETCH))).toBe(N * N);
  });
});

describe("cutStats on a large mesh", () => {
  it("counts removed and feathered quads", () => {
    const depth = make((x) => (x < 8 ? 2 : 6));
    const s = cutStats(depth, noSky, geom, yaw(8), STRETCH);
    expect(s.total).toBe(N * N);
    expect(s.removed).toBe(N);
    expect(s.removedFraction).toBeCloseTo(1 / N, 12);
    expect(cutStats(depth, noSky, geom, identityPose(), STRETCH).removed).toBe(0);
  });
});

// ------------------------------------------------------------- generated fixtures (npm run fixtures)
describe("fixture scenes at the identity pose", () => {
  const dir = resolve(import.meta.dirname, "../fixtures");
  for (const scene of ["room", "sky_tree", "two_layers"]) {
    const base = resolve(dir, scene);
    it.skipIf(!existsSync(resolve(base, "meta.json")))(`${scene}: under 0.5 % of quads cut`, () => {
      const meta = JSON.parse(readFileSync(resolve(base, "meta.json"), "utf-8")) as {
        preview_width: number;
        preview_height: number;
        f_px_preview: number;
      };
      const d = readFileSync(resolve(base, "depth.f16"));
      const depth = decodeHalfArray(new Uint16Array(d.buffer, d.byteOffset, d.byteLength / 2));
      const s = readFileSync(resolve(base, "sky.u8"));
      const g: MeshGeometry = {
        width: meta.preview_width,
        height: meta.preview_height,
        cols: 512,
        rows: Math.round((512 * meta.preview_height) / meta.preview_width),
        f: meta.f_px_preview,
      };
      const sky = new Uint8Array(s.buffer, s.byteOffset, s.byteLength);
      const adaptive = cutStats(depth, sky, g, identityPose(), STRETCH);
      const legacy = cutStats(depth, sky, g, identityPose(), TAU);
      expect(adaptive.removedFraction).toBeLessThan(0.005);
      expect(adaptive.removed).toBe(0);
      expect(legacy.removedFraction).toBeGreaterThanOrEqual(adaptive.removedFraction);
    });
  }
});
