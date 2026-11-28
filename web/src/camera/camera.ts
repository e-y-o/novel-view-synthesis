/**
 * Camera geometry: a pure TypeScript mirror of `src/nvs/core/camera.py` (no DOM, no WebGL).
 *
 * Conventions (OpenCV): x right, y down, z forward. The source camera is the identity at the origin. Pixel
 * centres sit at integer coordinates and the principal point is ((W-1)/2, (H-1)/2); fx = fy = `f` (pixels);
 * no distortion. A target camera is a pose `{R, t}` in the source frame: `P_target = Rᵀ (P − t)`, so `t` is
 * the target camera centre in metres and the **columns** of `R` are its right / down / forward axes.
 *
 * Matrices are plain row-major nested arrays (`Mat3`), the same shape as the JSON the server expects
 * (`R: [[..],[..],[..]]`). All maths runs in float64 (JS numbers).
 */

/** A 3-vector: metres for points and translations. */
export type Vec3 = [number, number, number];
/** One row of a `Mat3`. */
export type Row = [number, number, number];
/** A 3×3 matrix, row-major: `R[row][col]`. */
export type Mat3 = [Row, Row, Row];

/** A target camera in the source frame. `t` is in metres. */
export interface CameraSpec {
  R: Mat3;
  t: Vec3;
}

/**
 * Orbit controls around a pivot at `(0, 0, pivotZ)` in the source frame. No roll.
 *
 * - `yawDeg` > 0: the camera moves to the **right** and turns left; near content shifts **left** in the image
 *   relative to far content (parallax).
 * - `pitchDeg` > 0: the camera moves **up** (y is down, so `t.y < 0`) and tilts down; near content shifts
 *   **down** (larger v) relative to far content. Valid range is (−90°, 90°).
 * - `dolly` [m]: toward the pivot along the line of sight (positive = closer, negative = away), clamped to
 *   95 % of `pivotZ` so the camera never reaches the pivot.
 * - `pan` = [right, down] [m]: lateral shift along the camera's own x / y axes, applied last. The orientation
 *   is unchanged, so with `pan ≠ 0` the pivot leaves the principal point.
 * - `pivotZ` [m] > 0: the pivot depth.
 */
export interface OrbitParams {
  yawDeg: number;
  pitchDeg: number;
  dolly: number;
  pan: [number, number];
  pivotZ: number;
}

/** Orbit parameters as they appear in the server JSON (snake_case, README contract). */
export interface OrbitJson {
  yaw_deg: number;
  pitch_deg: number;
  dolly: number;
  pan: [number, number];
  pivot_z: number;
}

/** The pose payload of `POST /api/sessions/<id>/render` and `/coverage`. `orbit` is informational. */
export interface PoseJson {
  R: Mat3;
  t: Vec3;
  orbit?: OrbitJson;
}

/** Fraction of `pivotZ` the camera may approach the pivot by dollying. Same value as Python. */
export const DOLLY_MAX_FRACTION = 0.95;

const RAD = Math.PI / 180;

/** Pixel (u, v) [px] with z-depth `z` [m] to a camera-space point [X, Y, Z] [m]. */
export function unproject(u: number, v: number, z: number, f: number, W: number, H: number): Vec3 {
  const cx = (W - 1) / 2;
  const cy = (H - 1) / 2;
  return [((u - cx) / f) * z, ((v - cy) / f) * z, z];
}

/** Camera-space point (X, Y, Z) [m] to pixel [u, v] [px]. `Z <= 0` gives meaningless or infinite values. */
export function project(
  X: number,
  Y: number,
  Z: number,
  f: number,
  W: number,
  H: number,
): [number, number] {
  const cx = (W - 1) / 2;
  const cy = (H - 1) / 2;
  return [(f * X) / Z + cx, (f * Y) / Z + cy];
}

/** Source-frame point `P` [m] to the target camera frame: `Rᵀ (P − t)`. */
export function transformToTarget(P: Vec3, R: Mat3, t: Vec3): Vec3 {
  const d0 = P[0] - t[0];
  const d1 = P[1] - t[1];
  const d2 = P[2] - t[2];
  return [
    R[0][0] * d0 + R[1][0] * d1 + R[2][0] * d2,
    R[0][1] * d0 + R[1][1] * d1 + R[2][1] * d2,
    R[0][2] * d0 + R[1][2] * d1 + R[2][2] * d2,
  ];
}

/** The identity pose: the source camera itself. */
export function identityPose(): CameraSpec {
  return {
    R: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    t: [0, 0, 0],
  };
}

/** Orbit parameters that give the identity pose for a pivot at `pivotZ` metres. */
export function defaultOrbit(pivotZ: number): OrbitParams {
  return { yawDeg: 0, pitchDeg: 0, dolly: 0, pan: [0, 0], pivotZ };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * Pose of a camera orbiting the pivot and looking at it. Identical to Python's `orbit_pose` (float64 here,
 * float32 there). All zeros gives the identity.
 *
 * @throws RangeError for non-finite input, `pivotZ <= 0` or a pitch of ±90° (no defined right axis).
 */
export function orbitPose(p: OrbitParams): CameraSpec {
  const { yawDeg, pitchDeg, dolly, pan, pivotZ } = p;
  if (![yawDeg, pitchDeg, dolly, pan[0], pan[1], pivotZ].every(Number.isFinite)) {
    throw new RangeError("orbit parameters must be finite numbers");
  }
  if (!(pivotZ > 0)) throw new RangeError("pivotZ must be positive");
  const yaw = yawDeg * RAD;
  const pitch = pitchDeg * RAD;
  if (Math.abs(Math.cos(pitch)) < 1e-9)
    throw new RangeError("pitch must be inside (-90, 90) degrees");

  const radius = pivotZ - Math.min(dolly, DOLLY_MAX_FRACTION * pivotZ);
  const offset: Vec3 = [
    radius * Math.sin(yaw) * Math.cos(pitch),
    -radius * Math.sin(pitch),
    -radius * Math.cos(yaw) * Math.cos(pitch),
  ];
  const fwd: Vec3 = [-offset[0] / radius, -offset[1] / radius, -offset[2] / radius];
  const rightRaw = cross([0, 1, 0], fwd);
  const n = Math.hypot(rightRaw[0], rightRaw[1], rightRaw[2]);
  const right: Vec3 = [rightRaw[0] / n, rightRaw[1] / n, rightRaw[2] / n];
  const down = cross(fwd, right);

  const t: Vec3 = [
    offset[0] + pan[0] * right[0] + pan[1] * down[0],
    offset[1] + pan[0] * right[1] + pan[1] * down[1],
    pivotZ + offset[2] + pan[0] * right[2] + pan[1] * down[2],
  ];
  // columns of R are right, down, forward
  const R: Mat3 = [
    [right[0], down[0], fwd[0]],
    [right[1], down[1], fwd[1]],
    [right[2], down[2], fwd[2]],
  ];
  return { R, t };
}

/** Server JSON form of orbit parameters. */
export function orbitToJson(p: OrbitParams): OrbitJson {
  return {
    yaw_deg: p.yawDeg,
    pitch_deg: p.pitchDeg,
    dolly: p.dolly,
    pan: [p.pan[0], p.pan[1]],
    pivot_z: p.pivotZ,
  };
}

/** Inverse of `orbitToJson`. */
export function orbitFromJson(j: OrbitJson): OrbitParams {
  return {
    yawDeg: j.yaw_deg,
    pitchDeg: j.pitch_deg,
    dolly: j.dolly,
    pan: [j.pan[0], j.pan[1]],
    pivotZ: j.pivot_z,
  };
}

/**
 * The request body for the pose endpoints: `{R, t, orbit}`. `R` and `t` are authoritative, `orbit` only
 * informs the UI and the EXIF note. Deep-copies, so later mutation of the pose does not change the payload.
 */
export function poseToJson(pose: CameraSpec, orbit?: OrbitParams): PoseJson {
  const json: PoseJson = {
    R: [[...pose.R[0]], [...pose.R[1]], [...pose.R[2]]],
    t: [...pose.t],
  };
  if (orbit) json.orbit = orbitToJson(orbit);
  return json;
}

/**
 * Largest orbit angle the user may reach in each direction before the view would show too many holes
 * (degrees, from the server's `/limit`). `Infinity` = unlimited. `yawPlus` is the limit for `yawDeg > 0`.
 */
export interface OrbitLimits {
  yawPlusDeg: number;
  yawMinusDeg: number;
  pitchPlusDeg: number;
  pitchMinusDeg: number;
}

/** Where the rubber band starts to resist, as a fraction of the limit `A`. */
export const RUBBER_KNEE = 0.7;

/**
 * Rubber-band resistance: the angle shown for a requested magnitude `x >= 0` against a limit `A`.
 * Identity up to the knee `k·A` (`k = RUBBER_KNEE`), then a smooth exponential approach to `A`:
 * `kA + (1-k)A · (1 - exp(-(x - kA) / ((1-k)A)))`. Continuous, strictly increasing, with slope 1 at the knee
 * (no kink) and never reaching `A`: a soft wall, not a hard stop. `A <= 0` gives 0; `A = Infinity` the identity.
 */
export function rubberBand(x: number, limit: number): number {
  if (!(x > 0)) return x;
  if (limit === Infinity) return x;
  if (!(limit > 0)) return 0;
  const knee = RUBBER_KNEE * limit;
  if (x <= knee) return x;
  const span = limit - knee;
  return knee + span * (1 - Math.exp(-(x - knee) / span));
}

/**
 * Soft ("rubber band") clamp of the orbit angles to `limits`: yaw and pitch go through `rubberBand` in their own
 * direction, everything else is untouched. Pure; used as the controller's soft limit.
 */
export function clampOrbit(params: OrbitParams, limits: OrbitLimits): OrbitParams {
  const yaw =
    params.yawDeg >= 0
      ? rubberBand(params.yawDeg, limits.yawPlusDeg)
      : -rubberBand(-params.yawDeg, limits.yawMinusDeg);
  const pitch =
    params.pitchDeg >= 0
      ? rubberBand(params.pitchDeg, limits.pitchPlusDeg)
      : -rubberBand(-params.pitchDeg, limits.pitchMinusDeg);
  return { ...params, yawDeg: yaw, pitchDeg: pitch, pan: [params.pan[0], params.pan[1]] };
}
