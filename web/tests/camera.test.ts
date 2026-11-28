import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  choosePivot,
  clampOrbit,
  decodeHalfArray,
  defaultOrbit,
  floatToHalf,
  halfToFloat,
  identityPose,
  orbitFromJson,
  orbitPose,
  orbitToJson,
  poseToJson,
  project,
  transformToTarget,
  unproject,
  type Mat3,
  type OrbitJson,
  type OrbitParams,
  type Vec3,
} from "../src/camera";

// ------------------------------------------------------------------------------------- golden vectors
interface GoldenPixel {
  u: number;
  v: number;
  z: number;
  u_t: number;
  v_t: number;
  z_t: number;
}
interface GoldenCase {
  id: number;
  width: number;
  height: number;
  f_px: number;
  orbit?: OrbitJson;
  R: Mat3;
  t: Vec3;
  pixels: GoldenPixel[];
}
const golden = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../../tests/golden/camera_vectors.json"), "utf-8"),
) as { cases: GoldenCase[] };

describe("golden vectors (shared with Python)", () => {
  it("has the 36 cases: 24 orbits and 12 free poses", () => {
    expect(golden.cases).toHaveLength(36);
    expect(golden.cases.filter((c) => c.orbit).length).toBe(24);
  });

  for (const c of golden.cases) {
    it(`case ${c.id}: pose and pixel projections${c.orbit ? " (orbit)" : ""}`, () => {
      let R = c.R;
      let t = c.t;
      if (c.orbit) {
        const pose = orbitPose(orbitFromJson(c.orbit));
        // the JSON holds float32 values, so allow float32 rounding
        for (let r = 0; r < 3; r += 1)
          for (let k = 0; k < 3; k += 1)
            expect(Math.abs(pose.R[r]![k]! - c.R[r]![k]!)).toBeLessThan(1e-5);
        for (let k = 0; k < 3; k += 1) expect(Math.abs(pose.t[k]! - c.t[k]!)).toBeLessThan(1e-5);
        R = pose.R;
        t = pose.t;
      }
      for (const px of c.pixels) {
        const P = unproject(px.u, px.v, px.z, c.f_px, c.width, c.height);
        const Q = transformToTarget(P, R, t);
        const [u, v] = project(Q[0], Q[1], Q[2], c.f_px, c.width, c.height);
        expect(Math.abs(u - px.u_t)).toBeLessThan(1e-3);
        expect(Math.abs(v - px.v_t)).toBeLessThan(1e-3);
        expect(Math.abs(Q[2] - px.z_t)).toBeLessThan(1e-3);
      }
    });
  }

  it("with the JSON's own R and t the projections match to float64 precision", () => {
    for (const c of golden.cases) {
      for (const px of c.pixels) {
        const Q = transformToTarget(
          unproject(px.u, px.v, px.z, c.f_px, c.width, c.height),
          c.R,
          c.t,
        );
        const [u, v] = project(Q[0], Q[1], Q[2], c.f_px, c.width, c.height);
        expect(Math.abs(u - px.u_t)).toBeLessThan(1e-9);
        expect(Math.abs(v - px.v_t)).toBeLessThan(1e-9);
      }
    }
  });
});

// ------------------------------------------------------------------------------------------ geometry
const mulT = (R: Mat3): Mat3 => {
  // RᵀR
  const out: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i += 1)
    for (let j = 0; j < 3; j += 1)
      for (let k = 0; k < 3; k += 1) out[i]![j]! += R[k]![i]! * R[k]![j]!;
  return out;
};
const det = (R: Mat3): number =>
  R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
  R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
  R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);

const sample = (n: number): OrbitParams[] => {
  let s = 12345;
  const rnd = (lo: number, hi: number): number => {
    s = (s * 1103515245 + 12345) % 2 ** 31;
    return lo + ((hi - lo) * s) / 2 ** 31;
  };
  return Array.from({ length: n }, () => ({
    yawDeg: rnd(-90, 90),
    pitchDeg: rnd(-60, 60),
    dolly: rnd(-2, 4),
    pan: [rnd(-0.5, 0.5), rnd(-0.5, 0.5)] as [number, number],
    pivotZ: rnd(0.8, 12),
  }));
};

describe("projection", () => {
  it("round-trips project(unproject(...)) to 1e-9", () => {
    for (const [W, H, f] of [
      [640, 480, 500],
      [301, 203, 123.4],
      [4000, 3000, 5000],
    ] as const) {
      for (let i = 0; i < 50; i += 1) {
        const u = (((i * 7919) % 1000) / 1000) * (W - 1);
        const v = (((i * 104729) % 1000) / 1000) * (H - 1);
        const z = 0.5 + i * 0.7;
        const [X, Y, Z] = unproject(u, v, z, f, W, H);
        const [u2, v2] = project(X, Y, Z, f, W, H);
        expect(Math.abs(u2 - u)).toBeLessThan(1e-9);
        expect(Math.abs(v2 - v)).toBeLessThan(1e-9);
        expect(Z).toBe(z);
      }
    }
  });

  it("puts the principal point at ((W-1)/2, (H-1)/2)", () => {
    expect(project(0, 0, 3, 800, 641, 481)).toEqual([320, 240]);
    expect(unproject(320, 240, 3, 800, 641, 481)).toEqual([0, 0, 3]);
    expect(project(0, 0, 3, 800, 640, 480)).toEqual([319.5, 239.5]);
  });

  it("transformToTarget with the identity pose is a no-op, with a shift it moves the point", () => {
    const { R, t } = identityPose();
    expect(transformToTarget([1, 2, 3], R, t)).toEqual([1, 2, 3]);
    expect(transformToTarget([1, 2, 3], R, [1, 1, 1])).toEqual([0, 1, 2]);
  });
});

describe("orbitPose", () => {
  it("is the identity for all zeros", () => {
    const { R, t } = orbitPose(defaultOrbit(3.2));
    for (let r = 0; r < 3; r += 1)
      for (let c = 0; c < 3; c += 1) expect(R[r]![c]!).toBeCloseTo(r === c ? 1 : 0, 12);
    for (const v of t) expect(Math.abs(v)).toBeLessThan(1e-12);
  });

  it("always returns a proper rotation", () => {
    for (const p of sample(100)) {
      const { R } = orbitPose(p);
      const m = mulT(R);
      for (let i = 0; i < 3; i += 1)
        for (let j = 0; j < 3; j += 1) expect(m[i]![j]!).toBeCloseTo(i === j ? 1 : 0, 12);
      expect(det(R)).toBeCloseTo(1, 12);
    }
  });

  it("keeps the pivot at the principal point when pan is zero", () => {
    const W = 1001;
    const H = 751;
    for (const p of sample(60)) {
      const { R, t } = orbitPose({ ...p, pan: [0, 0] });
      const Q = transformToTarget([0, 0, p.pivotZ], R, t);
      const [u, v] = project(Q[0], Q[1], Q[2], 900, W, H);
      expect(Math.abs(u - (W - 1) / 2)).toBeLessThan(1e-9);
      expect(Math.abs(v - (H - 1) / 2)).toBeLessThan(1e-9);
      const radius = p.pivotZ - Math.min(p.dolly, 0.95 * p.pivotZ);
      expect(Q[2]).toBeCloseTo(radius, 9); // the pivot is straight ahead at the orbit radius
    }
  });

  it("moves the pivot off the principal point when panning", () => {
    const base = defaultOrbit(3);
    const { R, t } = orbitPose({ ...base, pan: [0.3, 0] });
    const Q = transformToTarget([0, 0, 3], R, t);
    const [u] = project(Q[0], Q[1], Q[2], 900, 1001, 751);
    expect(u).toBeLessThan(500 - 10); // camera moved right, so the pivot appears to the left
  });

  it("yaw +10 moves the camera right and shifts near content left relative to far content", () => {
    const pivotZ = 4;
    const { R, t } = orbitPose({ ...defaultOrbit(pivotZ), yawDeg: 10 });
    expect(t[0]).toBeGreaterThan(0);
    const near: Vec3 = [0, 0, pivotZ - 1.5];
    const far: Vec3 = [0, 0, pivotZ + 3];
    const uOf = (P: Vec3): number => {
      const Q = transformToTarget(P, R, t);
      return project(Q[0], Q[1], Q[2], 800, 801, 601)[0];
    };
    expect(uOf(near)).toBeLessThan(uOf(far));
  });

  it("pitch +10 moves the camera up (t.y < 0) and shifts near content down relative to far", () => {
    const pivotZ = 4;
    const { R, t } = orbitPose({ ...defaultOrbit(pivotZ), pitchDeg: 10 });
    expect(t[1]).toBeLessThan(0);
    const vOf = (P: Vec3): number => {
      const Q = transformToTarget(P, R, t);
      return project(Q[0], Q[1], Q[2], 800, 801, 601)[1];
    };
    expect(vOf([0, 0, pivotZ - 1.5])).toBeGreaterThan(vOf([0, 0, pivotZ + 3]));
  });

  it("clamps dolly to 95 % of the pivot depth and allows negative dolly", () => {
    const close = orbitPose({ ...defaultOrbit(2), dolly: 100 });
    expect(Math.hypot(close.t[0], close.t[1], close.t[2] - 2)).toBeCloseTo(0.05 * 2, 12);
    const far = orbitPose({ ...defaultOrbit(2), dolly: -3, yawDeg: 20 });
    expect(Math.hypot(far.t[0], far.t[1], far.t[2] - 2)).toBeCloseTo(5, 12);
  });

  it("rejects non-finite input, a non-positive pivot and a vertical pitch", () => {
    const ok = defaultOrbit(3);
    expect(() => orbitPose({ ...ok, yawDeg: NaN })).toThrow(RangeError);
    expect(() => orbitPose({ ...ok, dolly: Infinity })).toThrow(RangeError);
    expect(() => orbitPose({ ...ok, pan: [0, NaN] })).toThrow(RangeError);
    expect(() => orbitPose({ ...ok, pivotZ: 0 })).toThrow(RangeError);
    expect(() => orbitPose({ ...ok, pivotZ: -1 })).toThrow(RangeError);
    expect(() => orbitPose({ ...ok, pitchDeg: 90 })).toThrow(RangeError);
    expect(() => orbitPose({ ...ok, pitchDeg: -90 })).toThrow(RangeError);
    expect(() => orbitPose({ ...ok, pitchDeg: 89.9 })).not.toThrow();
  });
});

describe("JSON contract", () => {
  it("serialises {R, t, orbit} with the README field names", () => {
    const orbit: OrbitParams = {
      yawDeg: 5,
      pitchDeg: -2,
      dolly: 0.3,
      pan: [0.1, -0.2],
      pivotZ: 3.2,
    };
    const json = poseToJson(orbitPose(orbit), orbit);
    const parsed = JSON.parse(JSON.stringify(json)) as Record<string, unknown>;
    expect(Object.keys(parsed).sort()).toEqual(["R", "orbit", "t"]);
    expect(parsed.orbit).toEqual({
      yaw_deg: 5,
      pitch_deg: -2,
      dolly: 0.3,
      pan: [0.1, -0.2],
      pivot_z: 3.2,
    });
    expect((parsed.R as number[][]).map((r) => r.length)).toEqual([3, 3, 3]);
    expect((parsed.t as number[]).length).toBe(3);
    expect(orbitFromJson(orbitToJson(orbit))).toEqual(orbit);
  });

  it("omits orbit when not given and deep-copies the pose", () => {
    const pose = identityPose();
    const json = poseToJson(pose);
    expect("orbit" in json).toBe(false);
    pose.R[0][0] = 9;
    pose.t[0] = 9;
    expect(json.R[0][0]).toBe(1);
    expect(json.t[0]).toBe(0);
  });

  it("clampOrbit is an identity stub until T13", () => {
    const p: OrbitParams = { yawDeg: 99, pitchDeg: 5, dolly: 1, pan: [1, 2], pivotZ: 3 };
    const out = clampOrbit(p, { maxYawDeg: 10, maxPitchDeg: 10, maxDolly: 1, maxPan: 1 });
    expect(out).toEqual(p);
    expect(out.pan).not.toBe(p.pan);
  });
});

// --------------------------------------------------------------------------------------- half floats
describe("halfToFloat", () => {
  const table: [number, number][] = [
    [0x0000, 0],
    [0x0001, 2 ** -24], // smallest subnormal
    [0x03ff, (1023 / 1024) * 2 ** -14], // largest subnormal
    [0x0400, 2 ** -14], // smallest normal
    [0x3555, 0.333251953125],
    [0x3c00, 1],
    [0x4000, 2],
    [0x4900, 10],
    [0x5640, 100],
    [0x7bff, 65504], // largest finite
    [0xc000, -2],
    [0xbc00, -1],
    [0x7c00, Infinity],
    [0xfc00, -Infinity],
  ];
  for (const [bits, value] of table) {
    it(`0x${bits.toString(16).padStart(4, "0")} -> ${value}`, () => {
      expect(halfToFloat(bits)).toBe(value);
    });
  }

  it("keeps the sign of zero and decodes NaN", () => {
    expect(Object.is(halfToFloat(0x8000), -0)).toBe(true);
    expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true);
    expect(Number.isNaN(halfToFloat(0x7c01))).toBe(true);
  });

  it("round-trips every one of the 65536 bit patterns through floatToHalf", () => {
    for (let bits = 0; bits < 0x10000; bits += 1) {
      const x = halfToFloat(bits);
      const back = floatToHalf(x);
      if (Number.isNaN(x)) expect(Number.isNaN(halfToFloat(back))).toBe(true);
      else expect(back).toBe(bits);
    }
  });

  it("encodes arbitrary floats with at most half a unit of rounding error (float32-exact inputs)", () => {
    for (let i = 0; i < 2000; i += 1) {
      const x = Math.fround(0.3 * 1.003 ** i); // 0.3 .. ~120 m, like depths
      const back = halfToFloat(floatToHalf(x));
      expect(Math.abs(back - x) / x).toBeLessThanOrEqual(2 ** -11 + 1e-12);
    }
    expect(floatToHalf(70000)).toBe(0x7c00); // overflow
    expect(floatToHalf(65519)).toBe(0x7bff); // rounds down to the max finite
    expect(floatToHalf(65520)).toBe(0x7c00); // halfway to the next step rounds to even = Inf
    expect(floatToHalf(2 ** -25)).toBe(0); // half of the smallest subnormal ties to even (0)
    expect(floatToHalf(1.5 * 2 ** -24)).toBe(2); // tie rounds to even
    expect(floatToHalf(-0)).toBe(0x8000);
  });

  it("decodes arrays", () => {
    const out = decodeHalfArray(new Uint16Array([0x3c00, 0x4000, 0x7bff]));
    expect(Array.from(out)).toEqual([1, 2, 65504]);
    expect(out).toBeInstanceOf(Float32Array);
  });
});

// -------------------------------------------------------------------------------------------- pivot
describe("choosePivot", () => {
  const noSky = (n: number): Uint8Array => new Uint8Array(n);

  it("takes the 20th percentile with linear interpolation (numpy default)", () => {
    // values 1..11: pos = 0.2 * 10 = 2 -> exactly 3
    const depth = Float32Array.from({ length: 11 }, (_, i) => 11 - i); // unsorted
    expect(choosePivot(depth, noSky(11), 100)).toBe(3);
    // 1..6: pos = 1 -> 2 ; 1..4: pos = 0.6 -> 1 + 0.6 = 1.6
    expect(choosePivot(Float32Array.from([6, 5, 4, 3, 2, 1]), noSky(6), 100)).toBeCloseTo(2, 12);
    expect(choosePivot(Float32Array.from([1, 2, 3, 4]), noSky(4), 100)).toBeCloseTo(1.6, 12);
  });

  it("caps the result (default 5 m) and ignores sky, zeros and non-finite values", () => {
    const far = new Float32Array(100).fill(40);
    expect(choosePivot(far, noSky(100))).toBe(5);
    expect(choosePivot(far, noSky(100), 60)).toBe(40);
    const depth = Float32Array.from([2, 2, 2, 2, 2, 9, 9, 9, 0, NaN, Infinity, -1]);
    const sky = Uint8Array.from([0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0]);
    expect(choosePivot(depth, sky, 100)).toBe(2); // 9 m is sky: only the five 2 m pixels count
  });

  it("returns the cap when everything is sky or invalid", () => {
    expect(choosePivot(new Float32Array(10).fill(3), new Uint8Array(10).fill(255))).toBe(5);
    expect(choosePivot(new Float32Array(10).fill(3), new Uint8Array(10).fill(255), 8)).toBe(8);
    expect(choosePivot(new Float32Array(4).fill(NaN), noSky(4), 7)).toBe(7);
    expect(choosePivot(new Float32Array(0), new Uint8Array(0))).toBe(5);
  });

  it("decodes half-float depth (Uint16Array) and agrees with the float path", () => {
    const meters = Float32Array.from({ length: 5000 }, (_, i) => 0.5 + (i % 997) * 0.01);
    const half = Uint16Array.from(meters, (m) => floatToHalf(m));
    const sky = Uint8Array.from({ length: 5000 }, (_, i) => (i % 7 === 0 ? 255 : 0));
    const a = choosePivot(meters, sky, 100);
    const b = choosePivot(half, sky, 100);
    expect(Math.abs(a - b) / a).toBeLessThan(2 ** -10);
  });

  it("subsamples deterministically beyond a million valid pixels", () => {
    const n = 2_500_000;
    const depth = Float32Array.from({ length: n }, (_, i) => 1 + (i % 1000) / 100); // 1 .. 10.99
    const p = choosePivot(depth, noSky(n), 100);
    expect(p).toBeGreaterThan(2.9);
    expect(p).toBeLessThan(3.2); // true p20 is about 2.99
    expect(choosePivot(depth, noSky(n), 100)).toBe(p);
  });

  it("rejects mismatched lengths", () => {
    expect(() => choosePivot(new Float32Array(3), new Uint8Array(2))).toThrow(RangeError);
  });
});

// ----------------------------------------------------------------- against the server's own assets
describe("choosePivot on generated preview assets", () => {
  const dir = resolve(import.meta.dirname, "../fixtures");
  const scenes = existsSync(dir)
    ? readdirSync(dir).filter((name) => existsSync(resolve(dir, name, "meta.json")))
    : [];
  it.skipIf(scenes.length === 0)(
    "matches the server's pivot_z within 1 % (run `npm run fixtures`)",
    () => {
      for (const scene of scenes) {
        const meta = JSON.parse(readFileSync(resolve(dir, scene, "meta.json"), "utf-8")) as {
          pivot_z: number;
        };
        const d = readFileSync(resolve(dir, scene, "depth.f16"));
        const s = readFileSync(resolve(dir, scene, "sky.u8"));
        const depth = new Uint16Array(d.buffer, d.byteOffset, d.byteLength / 2);
        const pivot = choosePivot(depth, new Uint8Array(s.buffer, s.byteOffset, s.byteLength));
        expect(Math.abs(pivot - meta.pivot_z) / meta.pivot_z).toBeLessThan(0.01);
      }
    },
  );
});
