/**
 * Viewer-side logic of the coverage meter and the elastic limit.
 *
 * `ViewerLogic` is created with the camera controller of an open session. It feeds every pose change to the
 * latest-wins coverage loop, probes `/limit` along yaw and pitch once at start (and when the settings change)
 * to learn how far each direction may go, installs the rubber-band soft limit from those angles, and after the
 * interaction settles asks `/limit` for the exact answer and animates back inside it.
 */
import { signal } from "@preact/signals";
import {
  clampOrbit,
  orbitToJson,
  poseToJson,
  type CameraController,
  type CameraSpec,
  type OrbitLimits,
  type OrbitParams,
} from "../camera";
import {
  LatestWinsLoop,
  fetchCoverage,
  fetchLimit,
  isBenign,
  type Coverage,
  type LimitOptions,
  type LimitResult,
} from "../api/coverage";
import type { PoseJson } from "../camera";

/** Hole fraction up to which the result is "calm"; also the default limit. Shared with the meter. */
export const CALM_MAX = 0.12;
/** Above this, infill models start to invent objects (M0). */
export const WARN_MAX = 0.25;
export const MIN_LIMIT = 0.05;
export const MAX_LIMIT = 0.3;

export type Level = "calm" | "warn" | "danger";

export function coverageLevel(fraction: number): Level {
  if (fraction <= CALM_MAX) return "calm";
  if (fraction <= WARN_MAX) return "warn";
  return "danger";
}

/** The fraction the limit is about: all holes, or without the area outside the photo. */
export function countedFraction(c: Coverage, countOutOfFrame: boolean): number {
  return countOutOfFrame ? c.total : Math.max(0, c.total - c.out_of_frame);
}

export interface CoverageView {
  /** Last good value; never replaced by zeros while a request is pending. */
  value: Coverage | null;
  stale: boolean;
}

export interface LimitSettings {
  enabled: boolean;
  maxFraction: number;
  countOutOfFrame: boolean;
}

export const coverageView = signal<CoverageView>({ value: null, stale: false });
export const limitSettings = signal<LimitSettings>({
  enabled: true,
  maxFraction: CALM_MAX,
  countOutOfFrame: false,
});

let active: ViewerLogic | null = null;
function setActive(logic: ViewerLogic | null): void {
  active = logic;
}

/** Change the limit settings; the open viewer re-probes its limits. */
export function updateLimitSettings(patch: Partial<LimitSettings>): void {
  const next = { ...limitSettings.value, ...patch };
  next.maxFraction = Math.min(MAX_LIMIT, Math.max(MIN_LIMIT, next.maxFraction));
  limitSettings.value = next;
  active?.applySettings();
}

export interface ViewerDeps {
  fetchCoverage: typeof fetchCoverage;
  fetchLimit: typeof fetchLimit;
  frame?: (cb: () => void) => number;
  cancelFrame?: (id: number) => void;
  onError?: (e: unknown) => void;
}

/** At most about 30 coverage requests per second while dragging (one animation-frame pair at 60 Hz). */
export const COVERAGE_MIN_INTERVAL_MS = 33;

const PROBE_YAW = 45; // the controller's hard limits: how far the probes ask
const PROBE_PITCH = 30;
const LIMIT_ANIM_MS = 200;

export class ViewerLogic {
  private readonly loop: LatestWinsLoop<PoseJson, { coverage: Coverage; ms: number }>;
  private probeAbort: AbortController | null = null;
  private commitAbort: AbortController | null = null;
  private probeGeneration = 0;
  private disposed = false;
  /** Allowed angle per direction as last probed; `null` before the first probe finishes. */
  limits: OrbitLimits | null = null;

  constructor(
    private readonly id: string,
    private readonly controller: CameraController,
    private readonly deps: ViewerDeps = { fetchCoverage, fetchLimit },
  ) {
    setActive(this);
    coverageView.value = { value: null, stale: false };
    this.loop = new LatestWinsLoop({
      send: (pose, signal) => this.deps.fetchCoverage(this.id, pose, signal),
      onResult: (r) => {
        coverageView.value = { value: r.coverage, stale: false };
      },
      onStale: (stale) => {
        if (coverageView.value.stale !== stale)
          coverageView.value = { ...coverageView.value, stale };
      },
      onError: (e) => this.deps.onError?.(e),
      minIntervalMs: COVERAGE_MIN_INTERVAL_MS,
      frame: deps.frame,
      cancelFrame: deps.cancelFrame,
    });
    this.onChange(controller.orbit, controller.camera);
    void this.applySettings();
  }

  /** Every pose change: at most one coverage request in flight, newest pose wins. */
  onChange(orbit: OrbitParams, camera: CameraSpec): void {
    this.loop.update(poseToJson(camera, orbit));
  }

  private options(): LimitOptions {
    const s = limitSettings.value;
    return { maxHoleFraction: s.maxFraction, countOutOfFrame: s.countOutOfFrame };
  }

  /** (Re)install the soft limit from fresh probes, or remove it when limiting is off. */
  async applySettings(): Promise<void> {
    if (this.disposed) return;
    this.probeAbort?.abort();
    const generation = (this.probeGeneration += 1);
    if (!limitSettings.value.enabled) {
      this.limits = null;
      this.controller.setSoftLimit(null);
      return;
    }
    const abort = new AbortController();
    this.probeAbort = abort;
    const pivotZ = this.controller.orbit.pivotZ;
    const ask = async (yaw: number, pitch: number): Promise<LimitResult> =>
      this.deps.fetchLimit(
        this.id,
        orbitToJson({ yawDeg: yaw, pitchDeg: pitch, dolly: 0, pan: [0, 0], pivotZ }),
        this.options(),
        abort.signal,
      );
    try {
      // sequential: the worker runs one job at a time anyway, and the soft limit needs all four
      const yawPlus = await ask(PROBE_YAW, 0);
      const yawMinus = await ask(-PROBE_YAW, 0);
      const pitchPlus = await ask(0, PROBE_PITCH);
      const pitchMinus = await ask(0, -PROBE_PITCH);
      if (this.disposed || generation !== this.probeGeneration) return;
      this.limits = {
        yawPlusDeg: Math.abs(yawPlus.yaw_deg),
        yawMinusDeg: Math.abs(yawMinus.yaw_deg),
        pitchPlusDeg: Math.abs(pitchPlus.pitch_deg),
        pitchMinusDeg: Math.abs(pitchMinus.pitch_deg),
      };
      const limits = this.limits;
      this.controller.setSoftLimit((o) => clampOrbit(o, limits));
    } catch (e) {
      if (!isBenign(e)) this.deps.onError?.(e);
    }
  }

  /** Interaction settled: ask for the exact limit and animate back inside it if the view went too far. */
  async onCommit(orbit: OrbitParams): Promise<void> {
    if (this.disposed || !limitSettings.value.enabled) return;
    this.commitAbort?.abort();
    const abort = new AbortController();
    this.commitAbort = abort;
    try {
      const r = await this.deps.fetchLimit(
        this.id,
        orbitToJson(orbit),
        this.options(),
        abort.signal,
      );
      if (this.disposed || abort.signal.aborted || !r.limited) return;
      const now = this.controller.orbit;
      const moved =
        this.controller.moving ||
        now.yawDeg !== orbit.yawDeg ||
        now.pitchDeg !== orbit.pitchDeg ||
        now.dolly !== orbit.dolly;
      if (moved) return; // the user already moved on: the next commit decides
      this.controller.animateTo(
        {
          ...orbit,
          yawDeg: r.yaw_deg,
          pitchDeg: r.pitch_deg,
          dolly: r.dolly,
          pan: [r.pan[0], r.pan[1]],
        },
        LIMIT_ANIM_MS,
      );
    } catch (e) {
      if (!isBenign(e)) this.deps.onError?.(e);
    }
  }

  dispose(): void {
    this.disposed = true;
    this.loop.dispose();
    this.probeAbort?.abort();
    this.commitAbort?.abort();
    this.controller.setSoftLimit(null);
    if (active === this) setActive(null);
  }
}
