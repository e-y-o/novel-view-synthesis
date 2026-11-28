/**
 * Camera interaction logic (no DOM, no GL): turns gestures into orbit parameters and emits changes.
 *
 * Feel and conventions
 * - **Grab the world.** The scene follows the cursor: dragging right moves the content right, which means the
 *   camera moves left, so `yaw -= dx · degPerPx`; dragging down moves content down, `pitch += dy · degPerPx`
 *   (T08: pitch > 0 moves the camera up and shifts near content down). `degPerPx = 0.2 · 1000 / canvasWidth`.
 * - **Pan** moves the pivot-depth plane exactly with the cursor: metres per canvas pixel = `pivotZ / focalCanvasPx`,
 *   with the camera moving opposite to the drag (`pan = [right, down]` of the camera, T08).
 * - **Dolly** (wheel, pinch, keys) scales with `pivotZ`: a 100-unit wheel notch is `0.1 · pivotZ`; scrolling up
 *   (negative `deltaY`) moves closer.
 * - **Pivot** is always the point `(0, 0, pivotZ)` of the source frame (T08), so picking a point only chooses
 *   its depth. When the pivot changes the pose is recomputed so that the camera does not move at all (`yaw` and
 *   `pitch` are kept, `dolly` and `pan` absorb the difference): no animation is needed and nothing jumps.
 * - **Reset** animates (≈ 250 ms ease-out) to the identity pose, then restores the original pivot depth (the
 *   identity pose does not depend on the pivot).
 * - **Inertia**: after an orbit drag the last velocity decays exponentially (half-life 120 ms) and stops below
 *   0.01°/frame; off with reduced motion. Any new gesture cancels it.
 * - **Limits**: hard clamps `|yaw| ≤ 45°`, `|pitch| ≤ 30°`, `dolly ∈ [−1.5 pivot, 0.95 pivot]`, `|pan| ≤ pivot`.
 *   `setSoftLimit(fn)` lets T13 map the *requested* orbit to the *allowed* one (elastic limit); the controller
 *   keeps the requested orbit and publishes the allowed one.
 * - `onChange` fires at most once per animation frame; `onCommit` once after 200 ms without input or motion.
 *   The controller never draws: consumers call the renderer from `onChange`.
 */
import {
  DOLLY_MAX_FRACTION,
  defaultOrbit,
  orbitPose,
  type CameraSpec,
  type OrbitParams,
} from "./camera";

/** Time and scheduling, injectable so tests run on a fake clock. */
export interface Clock {
  now(): number;
  requestFrame(cb: (time: number) => void): number;
  cancelFrame(id: number): void;
  setTimeout(cb: () => void, ms: number): number;
  clearTimeout(id: number): void;
}

export const browserClock: Clock = {
  now: () => performance.now(),
  requestFrame: (cb) => requestAnimationFrame(cb),
  cancelFrame: (id) => cancelAnimationFrame(id),
  setTimeout: (cb, ms) => window.setTimeout(cb, ms),
  clearTimeout: (id) => window.clearTimeout(id),
};

export interface HardLimits {
  maxYawDeg: number;
  maxPitchDeg: number;
  /** Dolly lower bound as a multiple of the pivot depth (negative = away). */
  minDollyFactor: number;
}

export const DEFAULT_LIMITS: HardLimits = { maxYawDeg: 45, maxPitchDeg: 30, minDollyFactor: -1.5 };

/** Maps the requested orbit to the allowed one. Must be pure and fast (called every frame). */
export type SoftLimit = (requested: OrbitParams) => OrbitParams;

export interface ControllerOptions {
  pivotZ: number;
  onChange?: (orbit: OrbitParams, camera: CameraSpec) => void;
  /** Interaction settled (200 ms without input or motion): time for the expensive consumers. */
  onCommit?: (orbit: OrbitParams) => void;
  clock?: Clock;
  limits?: HardLimits;
  /** Default: the `prefers-reduced-motion` media query. */
  reducedMotion?: () => boolean;
  /** Called when pick-pivot mode turns on or off (a UI toggle mirrors it). */
  onPickMode?: (on: boolean) => void;
  /** Depth (metres, source frame) at a canvas pixel, or `null` for sky / unknown. Needed for pivot picking. */
  depthAt?: (canvasX: number, canvasY: number) => number | null;
}

const DEG_PER_PX_AT_1000 = 0.2;
const RESET_MS = 250;
const COMMIT_MS = 200;
const INERTIA_HALF_LIFE_MS = 120;
const INERTIA_STOP_DEG = 0.01; // per frame
const VELOCITY_WINDOW_MS = 100;
const WHEEL_DOLLY = 0.001; // of pivotZ per wheel unit
const KEY_STEP_DEG = 1;
const KEY_STEP_BIG_DEG = 5;
const KEY_DOLLY = 0.05; // of pivotZ

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
const easeOut = (t: number): number => 1 - (1 - t) ** 3;

function reducedMotionDefault(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function copy(o: OrbitParams): OrbitParams {
  return { ...o, pan: [o.pan[0], o.pan[1]] };
}

export class CameraController {
  private requested: OrbitParams;
  private allowed: OrbitParams;
  private readonly originalPivot: number;
  private readonly clock: Clock;
  private readonly limits: HardLimits;
  private readonly reducedMotion: () => boolean;
  private readonly onChange: ControllerOptions["onChange"];
  private readonly onCommit: ControllerOptions["onCommit"];
  private readonly onPickMode: ControllerOptions["onPickMode"];
  private depthAt: ControllerOptions["depthAt"];
  private softLimit: SoftLimit = (o) => o;
  private width = 1000;
  private focalPx = 1000;
  private frame = 0;
  private lastFrameTime = 0;
  private dirty = false;
  private commitTimer = 0;
  private samples: { t: number; yaw: number; pitch: number }[] = [];
  private velocity: { yaw: number; pitch: number } | null = null; // degrees per ms
  private anim: { from: OrbitParams; to: OrbitParams; start: number; ms: number } | null = null;
  private disposed = false;
  private pick = false;

  constructor(options: ControllerOptions) {
    this.requested = defaultOrbit(options.pivotZ);
    this.allowed = copy(this.requested);
    this.originalPivot = options.pivotZ;
    this.clock = options.clock ?? browserClock;
    this.limits = options.limits ?? DEFAULT_LIMITS;
    this.reducedMotion = options.reducedMotion ?? reducedMotionDefault;
    this.onChange = options.onChange;
    this.onCommit = options.onCommit;
    this.onPickMode = options.onPickMode;
    this.depthAt = options.depthAt;
  }

  // ------------------------------------------------------------------------------------------ state
  /** The allowed orbit (after hard clamps and the soft limit). */
  get orbit(): OrbitParams {
    return copy(this.allowed);
  }

  /** The orbit the user asked for (before the soft limit). */
  get requestedOrbit(): OrbitParams {
    return copy(this.requested);
  }

  get camera(): CameraSpec {
    return orbitPose(this.allowed);
  }

  /** True while an animation or inertia is running. */
  get moving(): boolean {
    return this.anim !== null || this.velocity !== null;
  }

  get pickMode(): boolean {
    return this.pick;
  }

  /** Canvas size and focal length in canvas pixels (`f_px_preview · fit.scale`), for the gesture scales. */
  setViewport(width: number, focalCanvasPx: number): void {
    if (width > 0) this.width = width;
    if (focalCanvasPx > 0) this.focalPx = focalCanvasPx;
  }

  setDepthSampler(fn: ControllerOptions["depthAt"]): void {
    this.depthAt = fn;
  }

  /** Install the elastic limit (T13). Takes effect immediately. */
  setSoftLimit(fn: SoftLimit | null): void {
    this.softLimit = fn ?? ((o) => o);
    this.apply();
  }

  /** Set the orbit programmatically (benchmarks, restoring a bookmark). Cancels motion. */
  setOrbit(orbit: OrbitParams): void {
    this.stopMotion();
    this.requested = copy(orbit);
    this.apply();
  }

  // ---------------------------------------------------------------------------------------- gestures
  /** A new gesture starts: cancels inertia and animations. */
  beginGesture(): void {
    this.stopMotion();
    this.samples = [];
  }

  /** Orbit drag by `dx, dy` canvas pixels (grab-the-world: see the module comment). */
  orbitDrag(dx: number, dy: number): void {
    const k = (DEG_PER_PX_AT_1000 * 1000) / this.width;
    this.requested.yawDeg -= dx * k;
    this.requested.pitchDeg += dy * k;
    this.clampRequested();
    this.samples.push({
      t: this.clock.now(),
      yaw: this.requested.yawDeg,
      pitch: this.requested.pitchDeg,
    });
    this.apply();
  }

  /** Pan drag: the pivot-depth plane follows the cursor. */
  panDrag(dx: number, dy: number): void {
    const m = this.requested.pivotZ / this.focalPx;
    this.requested.pan = [this.requested.pan[0] - dx * m, this.requested.pan[1] - dy * m];
    this.clampRequested();
    this.apply();
  }

  /** Dolly by wheel units (`deltaY` in pixels; negative = closer). */
  wheel(deltaY: number): void {
    this.dollyBy(-deltaY * WHEEL_DOLLY * this.requested.pivotZ);
  }

  /** Dolly by metres toward the pivot (negative = away). */
  dollyBy(metres: number): void {
    this.stopMotion();
    this.requested.dolly += metres;
    this.clampRequested();
    this.apply();
  }

  /** Pinch: `scale` > 1 is fingers spreading (closer). */
  pinch(scale: number): void {
    if (!(scale > 0)) return;
    this.dollyBy(Math.log(scale) * this.requested.pivotZ);
  }

  /** The orbit drag ended: start inertia from the last 100 ms of motion (unless reduced motion). */
  endOrbitGesture(): void {
    const now = this.clock.now();
    const recent = this.samples.filter((s) => now - s.t <= VELOCITY_WINDOW_MS);
    this.samples = [];
    const first = recent[0];
    const last = recent[recent.length - 1];
    if (!first || !last || recent.length < 2 || this.reducedMotion()) {
      this.settle();
      return;
    }
    const dt = last.t - first.t;
    if (dt <= 0) {
      this.settle();
      return;
    }
    const v = { yaw: (last.yaw - first.yaw) / dt, pitch: (last.pitch - first.pitch) / dt };
    if (Math.hypot(v.yaw, v.pitch) * 16 < INERTIA_STOP_DEG) {
      this.settle();
      return;
    }
    this.velocity = v;
    this.schedule();
  }

  /** A non-orbit gesture ended (pan, pinch). */
  endGesture(): void {
    this.samples = [];
    this.settle();
  }

  // ----------------------------------------------------------------------------------------- actions
  /** Animate back to the original view (identity pose, original pivot). Instant with reduced motion. */
  reset(): void {
    this.stopMotion();
    if (this.reducedMotion()) {
      this.requested = defaultOrbit(this.originalPivot);
      this.apply();
      return;
    }
    this.anim = {
      from: copy(this.requested),
      to: defaultOrbit(this.originalPivot),
      start: this.clock.now(),
      ms: RESET_MS,
    };
    this.schedule();
  }

  /**
   * Animate from the orbit shown now to `target` (ease-out, `ms` long). Starts from the *allowed* orbit, so the
   * elastic limit does not stall the start of the motion. Instant with reduced motion.
   */
  animateTo(target: OrbitParams, ms = 200): void {
    this.stopMotion();
    this.requested = copy(this.allowed);
    if (this.reducedMotion() || ms <= 0) {
      this.requested = copy(target);
      this.apply();
      return;
    }
    this.anim = { from: copy(this.requested), to: copy(target), start: this.clock.now(), ms };
    this.schedule();
  }

  /**
   * Make the point under canvas pixel `(x, y)` the orbit pivot (its depth becomes `pivotZ`). Returns false
   * for sky or when no depth sampler is set. The camera does not move.
   */
  pickPivotAt(x: number, y: number): boolean {
    const z = this.depthAt?.(x, y);
    this.setPick(false);
    if (z === null || z === undefined || !(z > 0) || !Number.isFinite(z)) return false;
    this.setPivot(z);
    return true;
  }

  /** Change the pivot depth keeping the camera pose exactly unchanged. */
  setPivot(pivotZ: number): void {
    if (!(pivotZ > 0)) throw new RangeError("pivotZ must be positive");
    this.stopMotion();
    this.requested = repivot(this.allowed, pivotZ);
    this.apply();
  }

  /** Toggle "pick pivot" mode (the next click or `pickPivotAt` call uses it). */
  togglePickMode(): boolean {
    this.setPick(!this.pick);
    return this.pick;
  }

  setPick(on: boolean): void {
    if (this.pick === on) return;
    this.pick = on;
    this.onPickMode?.(on);
  }

  /** Keyboard: returns true if the key was handled (the caller should `preventDefault`). */
  key(key: string, shift = false): boolean {
    const step = shift ? KEY_STEP_BIG_DEG : KEY_STEP_DEG;
    const k = this.width / 1000 / DEG_PER_PX_AT_1000; // pixels that equal one degree at this width
    switch (key) {
      case "ArrowLeft":
        this.keyOrbit(-step * k, 0);
        return true;
      case "ArrowRight":
        this.keyOrbit(step * k, 0);
        return true;
      case "ArrowUp":
        this.keyOrbit(0, -step * k);
        return true;
      case "ArrowDown":
        this.keyOrbit(0, step * k);
        return true;
      case "+":
      case "=":
        this.dollyBy(KEY_DOLLY * this.requested.pivotZ);
        this.settle();
        return true;
      case "-":
      case "_":
        this.dollyBy(-KEY_DOLLY * this.requested.pivotZ);
        this.settle();
        return true;
      case "0":
      case "Home":
        this.reset();
        return true;
      case "p":
      case "P":
        this.togglePickMode();
        return true;
      default:
        return false;
    }
  }

  private keyOrbit(dx: number, dy: number): void {
    this.stopMotion();
    this.orbitDrag(dx, dy);
    this.samples = [];
    this.settle();
  }

  // -------------------------------------------------------------------------------------- internals
  private clampRequested(): void {
    const r = this.requested;
    r.yawDeg = clamp(r.yawDeg, -this.limits.maxYawDeg, this.limits.maxYawDeg);
    r.pitchDeg = clamp(r.pitchDeg, -this.limits.maxPitchDeg, this.limits.maxPitchDeg);
    r.dolly = clamp(r.dolly, this.limits.minDollyFactor * r.pivotZ, DOLLY_MAX_FRACTION * r.pivotZ);
    r.pan = [clamp(r.pan[0], -r.pivotZ, r.pivotZ), clamp(r.pan[1], -r.pivotZ, r.pivotZ)];
  }

  /** Recompute the allowed orbit from the requested one and schedule an `onChange`. */
  private apply(): void {
    this.allowed = copy(this.softLimit(copy(this.requested)));
    this.dirty = true;
    this.schedule();
    this.touchCommit();
  }

  private stopMotion(): void {
    this.anim = null;
    this.velocity = null;
  }

  private settle(): void {
    this.touchCommit();
    this.schedule();
  }

  private schedule(): void {
    if (this.disposed || this.frame) return;
    this.frame = this.clock.requestFrame((t) => this.onFrame(t));
  }

  private touchCommit(): void {
    if (this.disposed) return;
    if (this.commitTimer) this.clock.clearTimeout(this.commitTimer);
    this.commitTimer = this.clock.setTimeout(() => {
      this.commitTimer = 0;
      if (this.moving) {
        this.touchCommit(); // still animating: wait for it to finish
        return;
      }
      this.onCommit?.(copy(this.allowed));
    }, COMMIT_MS);
  }

  private onFrame(time: number): void {
    this.frame = 0;
    if (this.disposed) return;
    const dt = this.lastFrameTime ? Math.min(time - this.lastFrameTime, 64) : 16;
    this.lastFrameTime = time;

    if (this.anim) {
      const { from: f, to } = this.anim;
      const t = clamp((time - this.anim.start) / this.anim.ms, 0, 1);
      const e = easeOut(t);
      const mix = (a: number, b: number): number => a + (b - a) * e;
      const r = this.requested;
      r.yawDeg = mix(f.yawDeg, to.yawDeg);
      r.pitchDeg = mix(f.pitchDeg, to.pitchDeg);
      r.dolly = mix(f.dolly, to.dolly);
      r.pan = [mix(f.pan[0], to.pan[0]), mix(f.pan[1], to.pan[1])];
      if (t >= 1) {
        this.requested = copy(to); // the pivot switches here: the identity pose does not depend on it
        this.anim = null;
      }
      this.dirty = true;
      this.allowed = copy(this.softLimit(copy(this.requested)));
      this.touchCommit();
    } else if (this.velocity) {
      const v = this.velocity;
      const decay = 0.5 ** (dt / INERTIA_HALF_LIFE_MS);
      v.yaw *= decay;
      v.pitch *= decay;
      if (Math.hypot(v.yaw, v.pitch) * dt < INERTIA_STOP_DEG) {
        this.velocity = null;
      } else {
        this.requested.yawDeg += v.yaw * dt;
        this.requested.pitchDeg += v.pitch * dt;
        const before = { y: this.requested.yawDeg, p: this.requested.pitchDeg };
        this.clampRequested();
        if (this.requested.yawDeg !== before.y) v.yaw = 0; // stop against a limit
        if (this.requested.pitchDeg !== before.p) v.pitch = 0;
        this.allowed = copy(this.softLimit(copy(this.requested)));
        this.dirty = true;
        this.touchCommit();
      }
    }

    if (this.dirty) {
      this.dirty = false;
      this.onChange?.(copy(this.allowed), orbitPose(this.allowed));
    }
    if (this.moving) this.schedule();
  }

  /** Cancel timers and callbacks; the controller ignores everything afterwards. */
  dispose(): void {
    this.disposed = true;
    if (this.frame) this.clock.cancelFrame(this.frame);
    if (this.commitTimer) this.clock.clearTimeout(this.commitTimer);
    this.frame = 0;
    this.commitTimer = 0;
    this.stopMotion();
  }
}

/**
 * Orbit parameters with pivot depth `pivotZ` that reproduce exactly the pose of `from` (same yaw and pitch, so
 * the same orientation; `dolly` and `pan` absorb the shift of the pivot). The orbit radius is clamped to at
 * least 5 % of the new pivot depth, so a pivot picked behind the camera still gives a valid orbit.
 */
export function repivot(from: OrbitParams, pivotZ: number): OrbitParams {
  const { R, t } = orbitPose(from);
  // columns of R: right, down, forward; the orbit offset direction is -forward
  const right: [number, number, number] = [R[0][0], R[1][0], R[2][0]];
  const down: [number, number, number] = [R[0][1], R[1][1], R[2][1]];
  const fwd: [number, number, number] = [R[0][2], R[1][2], R[2][2]];
  const d: [number, number, number] = [t[0], t[1], t[2] - pivotZ];
  const dot = (a: number[], b: number[]): number => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  const radius = Math.max(-dot(d, fwd), (1 - DOLLY_MAX_FRACTION) * pivotZ);
  return {
    yawDeg: from.yawDeg,
    pitchDeg: from.pitchDeg,
    dolly: pivotZ - radius,
    pan: [dot(d, right), dot(d, down)],
    pivotZ,
  };
}
