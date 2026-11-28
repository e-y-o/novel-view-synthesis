// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../src/api/client";
import { LatestWinsLoop, isBenign, type Coverage, type LimitResult } from "../src/api/coverage";
import {
  RUBBER_KNEE,
  clampOrbit,
  identityPose,
  orbitPose,
  rubberBand,
  type CameraController,
  type OrbitParams,
} from "../src/camera";
import {
  CALM_MAX,
  WARN_MAX,
  ViewerLogic,
  coverageLevel,
  coverageView,
  countedFraction,
  limitSettings,
  updateLimitSettings,
} from "../src/state/viewer";
import { CoverageMeter } from "../src/ui/CoverageMeter";

afterEach(cleanup);

// ---------------------------------------------------------------------------------------- rubber band
describe("rubberBand", () => {
  const A = 20;
  it("is the identity up to the knee and below", () => {
    for (const x of [0, 1, 5, 10, RUBBER_KNEE * A]) expect(rubberBand(x, A)).toBe(x);
  });

  it("is continuous and has slope 1 at the knee (no kink)", () => {
    const k = RUBBER_KNEE * A;
    const e = 1e-6;
    expect(rubberBand(k + e, A) - rubberBand(k, A)).toBeCloseTo(e, 9);
    const slopeAbove = (rubberBand(k + 2e-4, A) - rubberBand(k + 1e-4, A)) / 1e-4;
    expect(slopeAbove).toBeCloseTo(1, 3);
  });

  it("is strictly increasing, resists more and more, and never reaches the limit", () => {
    let prev = -1;
    let prevSlope = 2;
    for (let x = 0; x <= 60; x += 0.5) {
      const y = rubberBand(x, A);
      expect(y).toBeGreaterThan(prev);
      expect(y).toBeLessThan(A + 1e-9);
      if (x > 0) {
        const slope = y - rubberBand(x - 0.5, A);
        expect(slope).toBeLessThanOrEqual(prevSlope + 1e-12); // concave: the wall gets harder
        prevSlope = slope;
      }
      prev = y;
    }
    expect(rubberBand(1e4, A)).toBeCloseTo(A, 6); // asymptote
  });

  it("handles zero, infinite and negative inputs", () => {
    expect(rubberBand(10, 0)).toBe(0);
    expect(rubberBand(10, Infinity)).toBe(10);
    expect(rubberBand(0, A)).toBe(0);
    expect(rubberBand(-5, A)).toBe(-5); // callers pass magnitudes; negatives pass through
  });
});

describe("clampOrbit", () => {
  const base: OrbitParams = { yawDeg: 0, pitchDeg: 0, dolly: 0.3, pan: [0.1, 0.2], pivotZ: 3 };
  const limits = { yawPlusDeg: 10, yawMinusDeg: 20, pitchPlusDeg: 5, pitchMinusDeg: 8 };

  it("limits each direction on its own and keeps the rest", () => {
    const right = clampOrbit({ ...base, yawDeg: 45 }, limits);
    const left = clampOrbit({ ...base, yawDeg: -45 }, limits);
    expect(right.yawDeg).toBeGreaterThan(7);
    expect(right.yawDeg).toBeLessThan(10);
    expect(left.yawDeg).toBeLessThan(-14);
    expect(left.yawDeg).toBeGreaterThan(-20);
    expect(clampOrbit({ ...base, pitchDeg: 30 }, limits).pitchDeg).toBeLessThan(5);
    expect(clampOrbit({ ...base, pitchDeg: -30 }, limits).pitchDeg).toBeGreaterThan(-8);
    expect(right.dolly).toBe(0.3);
    expect(right.pan).toEqual([0.1, 0.2]);
    expect(right.pivotZ).toBe(3);
  });

  it("is odd-symmetric for equal limits and leaves small moves alone", () => {
    const sym = { yawPlusDeg: 12, yawMinusDeg: 12, pitchPlusDeg: 12, pitchMinusDeg: 12 };
    const a = clampOrbit({ ...base, yawDeg: 30, pitchDeg: 25 }, sym);
    const b = clampOrbit({ ...base, yawDeg: -30, pitchDeg: -25 }, sym);
    expect(b.yawDeg).toBeCloseTo(-a.yawDeg, 12);
    expect(b.pitchDeg).toBeCloseTo(-a.pitchDeg, 12);
    const small = clampOrbit({ ...base, yawDeg: 3, pitchDeg: -4 }, sym);
    expect(small.yawDeg).toBe(3);
    expect(small.pitchDeg).toBe(-4);
  });
});

// ---------------------------------------------------------------------------------- latest-wins loop
function loopHarness() {
  const frames: (() => void)[] = [];
  const calls: {
    payload: number;
    resolve: (v: number) => void;
    reject: (e: unknown) => void;
    signal: AbortSignal;
  }[] = [];
  let concurrent = 0;
  let maxConcurrent = 0;
  const results: number[] = [];
  const stale: boolean[] = [];
  const errors: unknown[] = [];
  const loop = new LatestWinsLoop<number, number>({
    send: (payload, signal) =>
      new Promise<number>((resolve, reject) => {
        concurrent += 1;
        maxConcurrent = Math.max(maxConcurrent, concurrent);
        calls.push({
          payload,
          signal,
          resolve: (v) => {
            concurrent -= 1;
            resolve(v);
          },
          reject: (e) => {
            concurrent -= 1;
            reject(e);
          },
        });
      }),
    onResult: (r) => results.push(r),
    onStale: (s) => stale.push(s),
    onError: (e) => errors.push(e),
    frame: (cb) => frames.push(cb),
    cancelFrame: () => undefined,
  });
  const runFrame = () => frames.splice(0).forEach((f) => f());
  const flush = () => new Promise((r) => setTimeout(r, 0));
  return { loop, calls, results, stale, errors, runFrame, flush, max: () => maxConcurrent, frames };
}

describe("LatestWinsLoop", () => {
  it("collapses 50 rapid updates into at most two requests and reflects the last one", async () => {
    const h = loopHarness();
    for (let i = 1; i <= 50; i += 1) h.loop.update(i);
    expect(h.frames.length).toBe(1); // one frame scheduled, not 50
    expect(h.calls.length).toBe(0); // nothing is sent before the frame
    h.runFrame();
    expect(h.calls.map((c) => c.payload)).toEqual([50]); // only the newest
    for (let i = 51; i <= 80; i += 1) h.loop.update(i); // more input while the first is in flight
    expect(h.calls.length).toBe(1); // still just one in flight
    h.calls[0]?.resolve(500);
    await h.flush();
    expect(h.results).toEqual([]); // that answer is already stale: dropped
    expect(h.calls.map((c) => c.payload)).toEqual([50, 80]); // the newest is sent at once, no frame wait
    h.calls[1]?.resolve(800);
    await h.flush();
    expect(h.results).toEqual([800]);
    expect(h.max()).toBe(1);
    expect(h.loop.sent).toBe(2);
    expect(h.stale.at(-1)).toBe(false);
  });

  it("keeps the old value marked stale until the newest answer arrives, never reporting zeros", async () => {
    const h = loopHarness();
    h.loop.update(1);
    h.runFrame();
    h.calls[0]?.resolve(10);
    await h.flush();
    expect(h.results).toEqual([10]);
    h.loop.update(2);
    expect(h.stale.at(-1)).toBe(true); // dimmed while the next one is on its way
    expect(h.results).toEqual([10]); // the last good value is untouched
    h.runFrame();
    h.calls[1]?.resolve(20);
    await h.flush();
    expect(h.results).toEqual([10, 20]);
    expect(h.stale.at(-1)).toBe(false);
  });

  it("stays quiet about superseded, cancelled and aborted requests but reports real errors", async () => {
    const h = loopHarness();
    h.loop.update(1);
    h.runFrame();
    h.calls[0]?.reject(new ApiError(409, "superseded", "newer"));
    await h.flush();
    h.loop.update(2);
    h.runFrame();
    h.calls[1]?.reject(new DOMException("aborted", "AbortError"));
    await h.flush();
    expect(h.errors).toEqual([]);
    h.loop.update(3);
    h.runFrame();
    h.calls[2]?.reject(new ApiError(503, "worker_died", "boom"));
    await h.flush();
    expect(h.errors).toHaveLength(1);
    expect(isBenign(new ApiError(409, "cancelled", ""))).toBe(true);
    expect(isBenign(new ApiError(500, "internal", ""))).toBe(false);
  });

  it("dispose aborts the request in flight and ignores everything after", async () => {
    const h = loopHarness();
    h.loop.update(1);
    h.runFrame();
    const call = h.calls[0]!;
    h.loop.dispose();
    expect(call.signal.aborted).toBe(true);
    call.resolve(1);
    await h.flush();
    expect(h.results).toEqual([]);
    h.loop.update(2);
    h.runFrame();
    expect(h.calls.length).toBe(1);
  });

  it("holds at most one request over a long drag at 120 updates per second", async () => {
    const h = loopHarness();
    let answered = 0;
    for (let tick = 0; tick < 300; tick += 1) {
      h.loop.update(tick);
      if (tick % 2 === 1) h.runFrame(); // a frame every other update
      if (tick % 3 === 0 && h.calls.length > answered) {
        h.calls[answered]?.resolve(tick);
        answered += 1;
        await h.flush();
      }
    }
    expect(h.max()).toBe(1);
    expect(h.loop.sent).toBeLessThan(300);
  });
});

// ------------------------------------------------------------------------------------------ thresholds
describe("coverage levels", () => {
  it("uses the named thresholds: calm up to 12 %, amber up to 25 %, red above", () => {
    expect([CALM_MAX, WARN_MAX]).toEqual([0.12, 0.25]);
    expect(coverageLevel(0)).toBe("calm");
    expect(coverageLevel(0.12)).toBe("calm");
    expect(coverageLevel(0.1201)).toBe("warn");
    expect(coverageLevel(0.25)).toBe("warn");
    expect(coverageLevel(0.2501)).toBe("danger");
  });

  it("counts the area outside the photo only when asked", () => {
    const c: Coverage = { disocclusion: 0.05, stretch: 0.01, out_of_frame: 0.2, total: 0.26 };
    expect(countedFraction(c, true)).toBe(0.26);
    expect(countedFraction(c, false)).toBeCloseTo(0.06, 12);
    expect(countedFraction({ ...c, total: 0.1, out_of_frame: 0.2 }, false)).toBe(0); // never negative
  });
});

// -------------------------------------------------------------------------------------------- the meter
describe("CoverageMeter", () => {
  const cov = (total: number, oof = 0): Coverage => ({
    disocclusion: total - oof,
    stretch: 0,
    out_of_frame: oof,
    total,
  });

  it("shows no numbers before the first answer", () => {
    render(<CoverageMeter coverage={null} stale={false} countOutOfFrame={false} />);
    expect(screen.queryByTestId("meter-total")).toBeNull();
    expect(screen.queryByRole("meter")).toBeNull();
  });

  it("says how much will be filled and rates it with text, not only colour", () => {
    const { rerender } = render(
      <CoverageMeter coverage={cov(0.06)} stale={false} countOutOfFrame={false} />,
    );
    expect(screen.getByTestId("meter-total").textContent).toBe("6.0 % will be filled");
    expect(screen.getByTestId("meter-level").textContent).toContain("Looks natural");
    rerender(<CoverageMeter coverage={cov(0.18)} stale={false} countOutOfFrame={false} />);
    expect(screen.getByTestId("meter-level").textContent).toContain("Quality may drop");
    expect(screen.getByTestId("meter-level").className).toContain("warn");
    rerender(<CoverageMeter coverage={cov(0.31)} stale={false} countOutOfFrame={false} />);
    expect(screen.getByTestId("meter-level").textContent).toContain("Large areas will be invented");
    expect(screen.getByTestId("meter-level").className).toContain("danger");
    const meter = screen.getByRole("meter");
    expect(meter.getAttribute("aria-valuenow")).toBe("31");
    expect(meter.getAttribute("aria-valuetext")).toContain("Large areas");
  });

  it("rates by the counted fraction: the outside-photo area is shown but only counts when asked", () => {
    const c: Coverage = { disocclusion: 0.03, stretch: 0, out_of_frame: 0.3, total: 0.33 };
    const { rerender } = render(
      <CoverageMeter coverage={c} stale={false} countOutOfFrame={false} />,
    );
    expect(screen.getByTestId("meter-total").textContent).toContain("33 %");
    expect(screen.getByTestId("meter-level").textContent).toContain("Looks natural");
    expect(screen.getByTestId("meter-level").textContent).toContain("not counted");
    rerender(<CoverageMeter coverage={c} stale={false} countOutOfFrame={true} />);
    expect(screen.getByTestId("meter-level").textContent).toContain("Large areas");
  });

  it("labels the three classes in a legend and marks stale values", () => {
    const { container } = render(
      <CoverageMeter
        coverage={{ disocclusion: 0.04, stretch: 0.02, out_of_frame: 0.05, total: 0.11 }}
        stale={true}
        countOutOfFrame={false}
      />,
    );
    expect(screen.getByText(/Hidden behind objects: 4\.0 %/)).toBeTruthy();
    expect(screen.getByText(/Stretched surfaces: 2\.0 %/)).toBeTruthy();
    expect(screen.getByText(/Outside the photo: 5\.0 %/)).toBeTruthy();
    expect(container.querySelector(".meter.stale")).not.toBeNull();
    expect(container.querySelectorAll(".seg")).toHaveLength(3);
  });
});

// -------------------------------------------------------------------------------------- viewer logic
type FakeController = Pick<
  CameraController,
  "orbit" | "camera" | "moving" | "setSoftLimit" | "animateTo"
>;

function fakeController(orbit: OrbitParams) {
  const state = { orbit, moving: false };
  const softLimits: (((o: OrbitParams) => OrbitParams) | null)[] = [];
  const animations: { target: OrbitParams; ms: number | undefined }[] = [];
  const controller: FakeController = {
    get orbit() {
      return state.orbit;
    },
    get camera() {
      return orbitPose(state.orbit);
    },
    get moving() {
      return state.moving;
    },
    setSoftLimit: (fn) => {
      softLimits.push(fn);
    },
    animateTo: (target, ms) => {
      animations.push({ target, ms });
    },
  };
  return { controller: controller as unknown as CameraController, state, softLimits, animations };
}

const limitResult = (over: Partial<LimitResult> = {}): LimitResult => ({
  scale: 1,
  yaw_deg: 0,
  pitch_deg: 0,
  dolly: 0,
  pan: [0, 0],
  hole_fraction: 0.05,
  limited: false,
  ...over,
});

describe("ViewerLogic", () => {
  const ORBIT: OrbitParams = { yawDeg: 0, pitchDeg: 0, dolly: 0, pan: [0, 0], pivotZ: 3 };
  beforeEach(() => {
    limitSettings.value = { enabled: true, maxFraction: 0.12, countOutOfFrame: false };
    coverageView.value = { value: null, stale: false };
  });

  function make(
    fetchLimit = vi.fn(async (_id: string, o: { yaw_deg: number; pitch_deg: number }) =>
      limitResult({
        yaw_deg: Math.sign(o.yaw_deg) * Math.min(Math.abs(o.yaw_deg), 12),
        pitch_deg: Math.sign(o.pitch_deg) * Math.min(Math.abs(o.pitch_deg), 6),
        limited: true,
      }),
    ),
  ) {
    const c = fakeController(ORBIT);
    const frames: (() => void)[] = [];
    const fetchCoverage = vi.fn(async () => ({
      coverage: { disocclusion: 0.04, stretch: 0, out_of_frame: 0.01, total: 0.05 },
      ms: 3,
    }));
    const logic = new ViewerLogic("a".repeat(64), c.controller, {
      fetchCoverage,
      fetchLimit: fetchLimit as never,
      frame: (cb) => frames.push(cb),
      cancelFrame: () => undefined,
    });
    return { ...c, logic, fetchCoverage, fetchLimit, frames };
  }
  const settle = () => new Promise((r) => setTimeout(r, 0));

  it("probes yaw and pitch in both directions at start and installs the rubber-band limit", async () => {
    const m = make();
    await settle();
    await settle();
    const calls = m.fetchLimit.mock.calls as unknown as [
      string,
      { yaw_deg: number; pitch_deg: number },
      { maxHoleFraction: number; countOutOfFrame: boolean },
    ][];
    expect(calls.map(([, o]) => [o.yaw_deg, o.pitch_deg])).toEqual([
      [45, 0],
      [-45, 0],
      [0, 30],
      [0, -30],
    ]);
    expect(calls[0]?.[2]).toEqual({ maxHoleFraction: 0.12, countOutOfFrame: false });
    expect(m.logic.limits).toEqual({
      yawPlusDeg: 12,
      yawMinusDeg: 12,
      pitchPlusDeg: 6,
      pitchMinusDeg: 6,
    });
    const soft = m.softLimits.at(-1);
    expect(soft).toBeTypeOf("function");
    const shown = soft!({ ...ORBIT, yawDeg: 40 });
    expect(shown.yawDeg).toBeGreaterThan(8.4); // past the knee: resisting
    expect(shown.yawDeg).toBeLessThan(12); // but never at the wall
    expect(soft!({ ...ORBIT, yawDeg: 5 }).yawDeg).toBe(5);
  });

  it("feeds pose changes to the coverage loop, one request at a time", async () => {
    const m = make();
    await settle();
    m.logic.onChange(ORBIT, orbitPose(ORBIT));
    m.logic.onChange({ ...ORBIT, yawDeg: 5 }, orbitPose({ ...ORBIT, yawDeg: 5 }));
    m.frames.splice(0).forEach((f) => f());
    await settle();
    await settle();
    expect(m.fetchCoverage.mock.calls.length).toBeLessThanOrEqual(2);
    expect(coverageView.value.value?.total).toBe(0.05);
    expect(coverageView.value.stale).toBe(false);
  });

  it("on commit, animates back to the limited orbit the server returns", async () => {
    const m = make();
    await settle();
    await settle();
    m.fetchLimit.mockClear();
    m.fetchLimit.mockResolvedValueOnce(
      limitResult({
        limited: true,
        scale: 0.5,
        yaw_deg: 10,
        pitch_deg: 2,
        dolly: 0.1,
        pan: [0.05, 0],
      }),
    );
    const far = {
      ...ORBIT,
      yawDeg: 20,
      pitchDeg: 4,
      dolly: 0.2,
      pan: [0.1, 0] as [number, number],
    };
    m.state.orbit = far;
    await m.logic.onCommit(far);
    expect(m.animations).toHaveLength(1);
    expect(m.animations[0]?.target).toEqual({
      ...far,
      yawDeg: 10,
      pitchDeg: 2,
      dolly: 0.1,
      pan: [0.05, 0],
    });
    expect(m.animations[0]?.ms).toBe(200);
  });

  it("does nothing on commit when the move is within the limit or the user moved on meanwhile", async () => {
    const m = make();
    await settle();
    await settle();
    m.fetchLimit.mockResolvedValueOnce(limitResult({ limited: false }));
    await m.logic.onCommit(ORBIT);
    expect(m.animations).toHaveLength(0);
    m.fetchLimit.mockImplementationOnce(async () => {
      m.state.orbit = { ...ORBIT, yawDeg: 9 }; // the user dragged again while the request ran
      return limitResult({ limited: true, yaw_deg: 1 });
    });
    await m.logic.onCommit(ORBIT);
    expect(m.animations).toHaveLength(0);
  });

  it("removes the soft limit and stops asking when limiting is switched off", async () => {
    const m = make();
    await settle();
    await settle();
    updateLimitSettings({ enabled: false });
    expect(m.softLimits.at(-1)).toBeNull();
    m.fetchLimit.mockClear();
    await m.logic.onCommit({ ...ORBIT, yawDeg: 30 });
    expect(m.fetchLimit).not.toHaveBeenCalled();
    updateLimitSettings({ enabled: true });
    await settle();
    await settle();
    expect(m.fetchLimit).toHaveBeenCalled();
  });

  it("re-probes with the new settings, clamping the slider range", async () => {
    const m = make();
    await settle();
    await settle();
    m.fetchLimit.mockClear();
    updateLimitSettings({ maxFraction: 0.9, countOutOfFrame: true });
    expect(limitSettings.value.maxFraction).toBe(0.3); // clamped to 30 %
    await settle();
    await settle();
    const first = m.fetchLimit.mock.calls[0] as unknown as [
      string,
      unknown,
      { maxHoleFraction: number; countOutOfFrame: boolean },
    ];
    expect(first[2]).toEqual({ maxHoleFraction: 0.3, countOutOfFrame: true });
    updateLimitSettings({ maxFraction: 0.01 });
    expect(limitSettings.value.maxFraction).toBe(0.05);
  });

  it("dispose cancels probes and removes the soft limit", async () => {
    const m = make();
    m.logic.dispose();
    expect(m.softLimits.at(-1)).toBeNull();
    await settle();
    m.logic.onChange(ORBIT, identityPose() as never);
    m.frames.splice(0).forEach((f) => f());
    await settle();
    expect(m.fetchCoverage.mock.calls.length).toBeLessThanOrEqual(1);
  });
});

describe("LatestWinsLoop rate cap", () => {
  it("never starts requests closer together than the interval, and still ends on the newest pose", async () => {
    let time = 0;
    const timers: { at: number; cb: () => void }[] = [];
    const frames: (() => void)[] = [];
    const starts: number[] = [];
    const payloads: number[] = [];
    const results: number[] = [];
    const loop = new LatestWinsLoop<number, number>({
      send: async (payload) => {
        starts.push(time);
        payloads.push(payload);
        return payload; // an instant server: only the rate cap can slow the loop down
      },
      onResult: (r) => results.push(r),
      frame: (cb) => frames.push(cb),
      cancelFrame: () => undefined,
      minIntervalMs: 33,
      now: () => time,
      setTimer: (cb, ms) => timers.push({ at: time + ms, cb }),
      clearTimer: () => undefined,
    });
    const settle = () => new Promise((r) => setTimeout(r, 0));
    for (let step = 0; step < 1000; step += 1) {
      time = step; // a pose update every millisecond for one second
      loop.update(step);
      frames.splice(0).forEach((f) => f());
      for (const t of timers.filter((t) => t.at <= time)) {
        timers.splice(timers.indexOf(t), 1);
        t.cb();
      }
      if (step % 5 === 0) await settle();
    }
    await settle();
    expect(starts.length).toBeLessThanOrEqual(Math.ceil(1000 / 33) + 1); // about 30 per second
    expect(starts.length).toBeGreaterThan(20);
    for (let i = 1; i < starts.length; i += 1)
      expect(starts[i]! - starts[i - 1]!).toBeGreaterThanOrEqual(33);
    // after the burst the last pose still arrives
    time = 2000;
    for (const t of timers.splice(0)) t.cb();
    await settle();
    expect(results.at(-1)).toBe(999);
  });
});
