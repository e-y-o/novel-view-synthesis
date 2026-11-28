import { describe, expect, it } from "vitest";
import {
  CameraController,
  attachControllerInput,
  floatToHalf,
  medianDepthAround,
  orbitPose,
  project,
  repivot,
  transformToTarget,
  type Clock,
  type InputTarget,
  type OrbitParams,
} from "../src/camera";

// ---------------------------------------------------------------------------------------- fake clock
class FakeClock implements Clock {
  time = 1000;
  private nextId = 1;
  private frames = new Map<number, (t: number) => void>();
  private timers = new Map<number, { at: number; cb: () => void }>();
  now(): number {
    return this.time;
  }
  requestFrame(cb: (t: number) => void): number {
    const id = this.nextId++;
    this.frames.set(id, cb);
    return id;
  }
  cancelFrame(id: number): void {
    this.frames.delete(id);
  }
  setTimeout(cb: () => void, ms: number): number {
    const id = this.nextId++;
    this.timers.set(id, { at: this.time + ms, cb });
    return id;
  }
  clearTimeout(id: number): void {
    this.timers.delete(id);
  }
  /** Advance by `ms` in 16 ms frames, running due timers and frame callbacks. */
  advance(ms: number): void {
    const end = this.time + ms;
    while (this.time < end) {
      this.time = Math.min(end, this.time + 16);
      for (const [id, t] of [...this.timers]) {
        if (t.at <= this.time) {
          this.timers.delete(id);
          t.cb();
        }
      }
      const due = [...this.frames];
      this.frames.clear();
      for (const [, cb] of due) cb(this.time);
    }
  }
  get pendingFrames(): number {
    return this.frames.size;
  }
}

interface Setup {
  clock: FakeClock;
  c: CameraController;
  changes: OrbitParams[];
  commits: OrbitParams[];
}
function setup(
  over: {
    reduced?: boolean;
    pivot?: number;
    depthAt?: (x: number, y: number) => number | null;
  } = {},
): Setup {
  const clock = new FakeClock();
  const changes: OrbitParams[] = [];
  const commits: OrbitParams[] = [];
  const c = new CameraController({
    pivotZ: over.pivot ?? 4,
    clock,
    reducedMotion: () => over.reduced ?? false,
    onChange: (o) => changes.push(o),
    onCommit: (o) => commits.push(o),
    depthAt: over.depthAt,
  });
  c.setViewport(1000, 1000);
  return { clock, c, changes, commits };
}

describe("orbit drag", () => {
  it("drags 100 px to 20 degrees at 1000 px wide, and the scene follows the cursor", () => {
    const { c } = setup();
    c.beginGesture();
    c.orbitDrag(-100, 0); // drag left: content moves left, the camera moves right
    expect(c.orbit.yawDeg).toBeCloseTo(20, 9);
    c.orbitDrag(200, 0); // drag right by 200
    expect(c.orbit.yawDeg).toBeCloseTo(-20, 9);
    c.orbitDrag(0, 50); // drag down: content moves down: pitch +10
    expect(c.orbit.pitchDeg).toBeCloseTo(10, 9);
    // the camera really moved opposite to the cursor
    expect(orbitPose({ ...c.orbit, yawDeg: 20, pitchDeg: 0 }).t[0]).toBeGreaterThan(0);
  });

  it("scales sensitivity with the canvas width", () => {
    const { c } = setup();
    c.setViewport(2000, 2000);
    c.orbitDrag(-100, 0);
    expect(c.orbit.yawDeg).toBeCloseTo(10, 9); // 0.2 * 1000 / 2000 per px
  });

  it("clamps to +-45 yaw and +-30 pitch", () => {
    const { c } = setup();
    c.orbitDrag(-1e5, 1e5);
    expect(c.orbit.yawDeg).toBe(45);
    expect(c.orbit.pitchDeg).toBe(30);
    c.orbitDrag(1e6, -1e6);
    expect(c.orbit.yawDeg).toBe(-45);
    expect(c.orbit.pitchDeg).toBe(-30);
  });
});

describe("pan and dolly", () => {
  it("pan follows the cursor exactly on the pivot plane and scales with the pivot depth", () => {
    for (const pivot of [2, 8]) {
      const { c } = setup({ pivot });
      c.setViewport(1000, 1250); // focal length in canvas pixels
      c.panDrag(100, 0); // drag right: the camera moves left
      const m = pivot / 1250;
      expect(c.orbit.pan[0]).toBeCloseTo(-100 * m, 12);
      // the pivot point (0, 0, pivot) moved right by exactly 100 canvas px
      const { R, t } = c.camera;
      const Q = transformToTarget([0, 0, pivot], R, t);
      const [u0] = project(0, 0, pivot, 1250, 1001, 1001);
      const [u1] = project(Q[0], Q[1], Q[2], 1250, 1001, 1001);
      expect(u1 - u0).toBeCloseTo(100, 6);
    }
  });

  it("wheel dolly scales with the pivot and clamps to [-1.5 pivot, 0.95 pivot]", () => {
    const a = setup({ pivot: 2 }).c;
    const b = setup({ pivot: 8 }).c;
    a.wheel(-100);
    b.wheel(-100);
    expect(a.orbit.dolly).toBeCloseTo(0.2, 12);
    expect(b.orbit.dolly).toBeCloseTo(0.8, 12);
    a.wheel(-1e6);
    expect(a.orbit.dolly).toBeCloseTo(0.95 * 2, 12);
    a.wheel(1e7);
    expect(a.orbit.dolly).toBeCloseTo(-1.5 * 2, 12);
  });

  it("pinch spreading moves closer, pinching in moves away", () => {
    const { c } = setup();
    c.pinch(1.5);
    expect(c.orbit.dolly).toBeGreaterThan(0);
    c.pinch(1 / 1.5);
    expect(c.orbit.dolly).toBeCloseTo(0, 12);
    c.pinch(-1); // ignored
    c.pinch(0);
    expect(c.orbit.dolly).toBeCloseTo(0, 12);
  });
});

describe("reset", () => {
  it("animates to the identity pose without overshoot and ends exactly there", () => {
    const { c, clock, changes } = setup();
    c.orbitDrag(-100, 40);
    c.wheel(-300);
    c.panDrag(30, -20);
    clock.advance(500); // let the commit timer pass
    changes.length = 0;
    const start = c.orbit;
    c.reset();
    clock.advance(300);
    const yaws = changes.map((o) => o.yawDeg);
    expect(yaws.length).toBeGreaterThan(5);
    for (let i = 1; i < yaws.length; i += 1) {
      expect(Math.abs(yaws[i]!)).toBeLessThanOrEqual(Math.abs(yaws[i - 1]!) + 1e-12); // monotonic
      expect(yaws[i]! * start.yawDeg).toBeGreaterThanOrEqual(0); // never crosses zero
    }
    const end = c.orbit;
    expect(end).toEqual({ yawDeg: 0, pitchDeg: 0, dolly: 0, pan: [0, 0], pivotZ: 4 });
    const pose = c.camera;
    expect(Math.abs(pose.t[0]) + Math.abs(pose.t[1]) + Math.abs(pose.t[2])).toBeLessThan(1e-9);
  });

  it("takes about 250 ms and is instant with reduced motion", () => {
    const a = setup();
    a.c.orbitDrag(-100, 0);
    a.clock.advance(300);
    a.c.reset();
    a.clock.advance(100);
    expect(a.c.orbit.yawDeg).not.toBe(0);
    a.clock.advance(200);
    expect(a.c.orbit.yawDeg).toBe(0);

    const b = setup({ reduced: true });
    b.c.orbitDrag(-100, 0);
    b.c.reset();
    expect(b.c.orbit.yawDeg).toBe(0);
  });

  it("restores the original pivot after a pivot change", () => {
    const { c, clock } = setup();
    c.setPivot(9);
    c.orbitDrag(-50, 0);
    c.reset();
    clock.advance(400);
    expect(c.orbit.pivotZ).toBe(4);
    expect(c.orbit.yawDeg).toBe(0);
  });
});

describe("inertia", () => {
  function fling(c: CameraController, clock: FakeClock): void {
    c.beginGesture();
    for (let i = 0; i < 6; i += 1) {
      clock.advance(16);
      c.orbitDrag(-10, 0);
    }
    c.endOrbitGesture();
  }

  it("continues after release, decays with a ~120 ms half-life and stops", () => {
    const { c, clock } = setup();
    fling(c, clock);
    const released = c.orbit.yawDeg;
    clock.advance(16 * 3);
    const early = c.orbit.yawDeg - released;
    expect(early).toBeGreaterThan(0); // still moving
    expect(c.moving).toBe(true);
    clock.advance(2000);
    expect(c.moving).toBe(false);
    const final = c.orbit.yawDeg;
    clock.advance(500);
    expect(c.orbit.yawDeg).toBe(final); // stopped
    // total coast distance of an exponential decay: v * halfLife / ln 2
    expect(final - released).toBeGreaterThan(early);
  });

  it("is disabled by reduced motion", () => {
    const { c, clock } = setup({ reduced: true });
    fling(c, clock);
    const released = c.orbit.yawDeg;
    clock.advance(1000);
    expect(c.orbit.yawDeg).toBe(released);
    expect(c.moving).toBe(false);
  });

  it("is cancelled by a new gesture, and a slow release does not coast", () => {
    const { c, clock } = setup();
    fling(c, clock);
    clock.advance(32);
    c.beginGesture(); // pointerdown mid-inertia
    expect(c.moving).toBe(false);
    const held = c.orbit.yawDeg;
    clock.advance(500);
    expect(c.orbit.yawDeg).toBe(held);

    c.beginGesture();
    c.orbitDrag(-1, 0);
    clock.advance(300); // finger rested before lifting: no velocity
    c.endOrbitGesture();
    expect(c.moving).toBe(false);
  });

  it("stops against a hard limit", () => {
    const { c, clock } = setup();
    c.beginGesture();
    for (let i = 0; i < 6; i += 1) {
      clock.advance(16);
      c.orbitDrag(-200, 0);
    }
    c.endOrbitGesture();
    clock.advance(3000);
    expect(c.orbit.yawDeg).toBe(45);
    expect(c.moving).toBe(false);
  });
});

describe("pivot", () => {
  const poses = (): OrbitParams[] => [
    { yawDeg: 17, pitchDeg: -8, dolly: 0.6, pan: [0.2, -0.1], pivotZ: 4 },
    { yawDeg: -30, pitchDeg: 25, dolly: -1, pan: [-0.3, 0.2], pivotZ: 2.5 },
    { yawDeg: 0, pitchDeg: 0, dolly: 0, pan: [0, 0], pivotZ: 6 },
  ];

  it("repivot keeps the camera pose unchanged (projected points move less than 0.1 px)", () => {
    for (const from of poses()) {
      for (const pivotZ of [1.2, 3, 9, 20]) {
        const to = repivot(from, pivotZ);
        expect(to.pivotZ).toBe(pivotZ);
        const a = orbitPose(from);
        const b = orbitPose(to);
        for (const P of [
          [0.3, -0.2, 2],
          [-1, 0.5, 6],
          [0, 0, 10],
        ] as const) {
          const qa = transformToTarget([...P], a.R, a.t);
          const qb = transformToTarget([...P], b.R, b.t);
          const [ua, va] = project(qa[0], qa[1], qa[2], 1500, 1000, 800);
          const [ub, vb] = project(qb[0], qb[1], qb[2], 1500, 1000, 800);
          expect(Math.hypot(ua - ub, va - vb)).toBeLessThan(0.1);
        }
      }
    }
  });

  it("pickPivotAt uses the sampled depth and ignores sky", () => {
    const { c, changes, clock } = setup({ depthAt: (x) => (x < 500 ? 7 : null) });
    c.orbitDrag(-100, 20);
    const before = c.camera;
    expect(c.pickPivotAt(100, 100)).toBe(true);
    expect(c.orbit.pivotZ).toBe(7);
    const after = c.camera;
    for (let i = 0; i < 3; i += 1) expect(after.t[i]).toBeCloseTo(before.t[i]!, 9);
    expect(c.pickPivotAt(900, 100)).toBe(false); // sky
    expect(c.orbit.pivotZ).toBe(7);
    clock.advance(40);
    expect(changes.length).toBeGreaterThan(0);
  });

  it("toggles pick mode, which a pick turns off", () => {
    const { c } = setup({ depthAt: () => 3 });
    expect(c.togglePickMode()).toBe(true);
    expect(c.pickMode).toBe(true);
    c.pickPivotAt(1, 1);
    expect(c.pickMode).toBe(false);
  });
});

describe("medianDepthAround", () => {
  it("takes the median of non-sky pixels in a 5x5 window", () => {
    const W = 8;
    const depth = Float32Array.from({ length: W * W }, (_, i) => 1 + (i % W)); // column x has depth 1 + x
    const sky = new Uint8Array(W * W);
    expect(medianDepthAround(depth, sky, W, W, 4, 4)).toBe(5);
    sky.fill(255);
    expect(medianDepthAround(depth, sky, W, W, 4, 4)).toBeNull();
    sky.fill(0);
    for (let y = 0; y < W; y += 1) for (let x = 0; x < 4; x += 1) sky[y * W + x] = 255; // left half sky
    // window x in 2..6 keeps 4,5,6: depths 5,6,7 -> median 6
    expect(medianDepthAround(depth, sky, W, W, 4, 4)).toBe(6);
  });

  it("decodes half floats and rejects outside coordinates and invalid depths", () => {
    const depth = Uint16Array.from({ length: 16 }, () => floatToHalf(2.5));
    const sky = new Uint8Array(16);
    expect(medianDepthAround(depth, sky, 4, 4, 1.4, 2.2)).toBeCloseTo(2.5, 3);
    expect(medianDepthAround(depth, sky, 4, 4, -1, 0)).toBeNull();
    expect(medianDepthAround(depth, sky, 4, 4, 0, 9)).toBeNull();
    expect(medianDepthAround(new Float32Array(16).fill(NaN), sky, 4, 4, 1, 1)).toBeNull();
  });
});

describe("events", () => {
  it("emits at most one onChange per frame under a burst of 100 events", () => {
    const { c, clock, changes } = setup();
    c.beginGesture();
    for (let i = 0; i < 100; i += 1) c.orbitDrag(-1, 0.5);
    expect(changes.length).toBe(0); // nothing synchronous
    clock.advance(16);
    expect(changes.length).toBe(1);
    expect(changes[0]!.yawDeg).toBeCloseTo(20, 9); // the last state, not 100 intermediate ones
    clock.advance(160);
    expect(changes.length).toBe(1);
  });

  it("fires onCommit once after 200 ms of no input, and not while input continues", () => {
    const { c, clock, commits } = setup();
    c.beginGesture();
    for (let i = 0; i < 10; i += 1) {
      c.orbitDrag(-5, 0);
      clock.advance(100); // below the settle time: no commit yet
    }
    expect(commits.length).toBe(0);
    c.endGesture();
    clock.advance(150);
    expect(commits.length).toBe(0);
    clock.advance(100);
    expect(commits.length).toBe(1);
    clock.advance(2000);
    expect(commits.length).toBe(1);
    expect(commits[0]!.yawDeg).toBeCloseTo(c.orbit.yawDeg, 12);
  });

  it("waits for inertia before committing", () => {
    const { c, clock, commits } = setup();
    c.beginGesture();
    for (let i = 0; i < 6; i += 1) {
      clock.advance(16);
      c.orbitDrag(-10, 0);
    }
    c.endOrbitGesture();
    clock.advance(200);
    expect(commits.length).toBe(0); // still coasting
    clock.advance(3000);
    expect(commits.length).toBe(1);
  });
});

describe("soft limit", () => {
  it("maps the requested orbit to the allowed one and keeps the request", () => {
    const { c, clock, changes } = setup();
    c.setSoftLimit((o) => ({ ...o, yawDeg: Math.max(-10, Math.min(10, o.yawDeg)) }));
    c.orbitDrag(-200, 0); // requested +40
    expect(c.requestedOrbit.yawDeg).toBeCloseTo(40, 9);
    expect(c.orbit.yawDeg).toBe(10);
    clock.advance(16);
    expect(changes.at(-1)!.yawDeg).toBe(10);
    c.setSoftLimit(null);
    expect(c.orbit.yawDeg).toBeCloseTo(40, 9);
  });
});

describe("keyboard", () => {
  it("arrows orbit by 1 degree (5 with shift), +/- dolly, 0 resets, p toggles pick", () => {
    const { c, clock } = setup();
    expect(c.key("ArrowRight")).toBe(true);
    expect(c.orbit.yawDeg).toBeCloseTo(-1, 9);
    c.key("ArrowLeft", true);
    expect(c.orbit.yawDeg).toBeCloseTo(4, 9);
    c.key("ArrowDown");
    expect(c.orbit.pitchDeg).toBeCloseTo(1, 9);
    c.key("ArrowUp", true);
    expect(c.orbit.pitchDeg).toBeCloseTo(-4, 9);
    c.key("+");
    expect(c.orbit.dolly).toBeCloseTo(0.2, 9);
    c.key("-");
    c.key("-");
    expect(c.orbit.dolly).toBeCloseTo(-0.2, 9);
    c.key("p");
    expect(c.pickMode).toBe(true);
    c.key("Home");
    clock.advance(400);
    expect(c.orbit.yawDeg).toBe(0);
    expect(c.key("q")).toBe(false);
  });
});

describe("dispose", () => {
  it("stops all callbacks", () => {
    const { c, clock, changes, commits } = setup();
    c.orbitDrag(-10, 0);
    c.dispose();
    clock.advance(1000);
    expect(changes.length).toBe(0);
    expect(commits.length).toBe(0);
    expect(clock.pendingFrames).toBe(0);
    c.orbitDrag(-10, 0);
    clock.advance(1000);
    expect(changes.length).toBe(0);
  });
});

// ------------------------------------------------------------------------------------- DOM wiring
class FakeElement implements InputTarget {
  listeners = new Map<string, Set<EventListener>>();
  style = { cursor: "", touchAction: "", userSelect: "" };
  tabIndex = -1;
  captured = new Set<number>();
  focused = false;
  addEventListener(type: string, fn: EventListener): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(fn);
  }
  removeEventListener(type: string, fn: EventListener): void {
    this.listeners.get(type)?.delete(fn);
  }
  setPointerCapture(id: number): void {
    this.captured.add(id);
  }
  releasePointerCapture(id: number): void {
    this.captured.delete(id);
  }
  focus(): void {
    this.focused = true;
  }
  getBoundingClientRect() {
    return { left: 10, top: 20, width: 1000, height: 800 };
  }
  fire(type: string, event: Record<string, unknown>): { prevented: boolean } {
    const e = { preventDefault: () => (state.prevented = true), ...event };
    const state = { prevented: false };
    for (const fn of [...(this.listeners.get(type) ?? [])]) fn(e as unknown as Event);
    return state;
  }
  get count(): number {
    return [...this.listeners.values()].reduce((n, s) => n + s.size, 0);
  }
}

const ptr = (id: number, x: number, y: number, extra: Record<string, unknown> = {}) => ({
  pointerId: id,
  clientX: x + 10,
  clientY: y + 20,
  button: 0,
  shiftKey: false,
  ...extra,
});

describe("DOM input", () => {
  function wired(over: Parameters<typeof setup>[0] = {}) {
    const s = setup(over);
    const el = new FakeElement();
    const detach = attachControllerInput(s.c, el, { pixelRatio: () => 1 });
    return { ...s, el, detach };
  }

  it("sets accessibility and gesture styles", () => {
    const { el } = wired();
    expect(el.tabIndex).toBe(0);
    expect(el.style.touchAction).toBe("none");
    expect(el.style.userSelect).toBe("none");
    expect(el.style.cursor).toBe("grab");
  });

  it("left drag orbits, with grab/grabbing cursors and capture", () => {
    const { el, c, clock } = wired();
    el.fire("pointerdown", ptr(1, 500, 400));
    expect(el.style.cursor).toBe("grabbing");
    expect(el.captured.has(1)).toBe(true);
    expect(el.focused).toBe(true);
    el.fire("pointermove", ptr(1, 400, 400)); // 100 px left
    expect(c.orbit.yawDeg).toBeCloseTo(20, 9);
    el.fire("pointerup", ptr(1, 400, 400));
    expect(el.style.cursor).toBe("grab");
    expect(el.captured.size).toBe(0);
    clock.advance(10);
  });

  it("right drag, middle drag and shift+left pan with a move cursor", () => {
    for (const extra of [{ button: 2 }, { button: 1 }, { shiftKey: true }]) {
      const { el, c } = wired();
      c.setViewport(1000, 1000);
      el.fire("pointerdown", ptr(1, 500, 400, extra));
      expect(el.style.cursor).toBe("move");
      el.fire("pointermove", ptr(1, 600, 400, extra));
      expect(c.orbit.pan[0]).toBeLessThan(0);
      expect(c.orbit.yawDeg).toBe(0);
      el.fire("pointerup", ptr(1, 600, 400, extra));
    }
  });

  it("two pointers pan and pinch", () => {
    const { el, c } = wired();
    el.fire("pointerdown", ptr(1, 400, 400));
    el.fire("pointerdown", ptr(2, 600, 400));
    el.fire("pointermove", ptr(2, 800, 400)); // spread 200 -> 400
    expect(c.orbit.dolly).toBeGreaterThan(0);
    const dolly = c.orbit.dolly;
    el.fire("pointermove", ptr(1, 500, 400)); // closer again, centroid moves right
    expect(c.orbit.dolly).toBeLessThan(dolly);
    el.fire("pointerup", ptr(2, 800, 400));
    el.fire("pointerup", ptr(1, 500, 400));
    expect(el.style.cursor).toBe("grab");
  });

  it("wheel dollies and prevents page scroll; ctrl+wheel is stronger", () => {
    const { el, c } = wired();
    const r = el.fire("wheel", { deltaY: -100, deltaMode: 0 });
    expect(r.prevented).toBe(true);
    expect(c.orbit.dolly).toBeCloseTo(0.4, 9);
    el.fire("wheel", { deltaY: 100, deltaMode: 0 });
    const r2 = el.fire("wheel", { deltaY: -10, deltaMode: 0, ctrlKey: true });
    expect(r2.prevented).toBe(true);
    expect(c.orbit.dolly).toBeCloseTo(0.4 * 0.3, 9);
  });

  it("double-click resets, alt+double-click picks the pivot under the cursor", () => {
    // client (500, 400) is canvas (490, 380): the element sits at (10, 20)
    const { el, c, clock } = wired({ depthAt: (x, y) => (x === 490 && y === 380 ? 6 : null) });
    el.fire("pointerdown", ptr(1, 500, 400));
    el.fire("pointermove", ptr(1, 400, 400));
    el.fire("pointerup", ptr(1, 400, 400));
    el.fire("dblclick", { clientX: 100, clientY: 100, altKey: true }); // sky or unknown: ignored
    expect(c.orbit.pivotZ).toBe(4);
    el.fire("dblclick", { clientX: 500, clientY: 400, altKey: false });
    clock.advance(500);
    expect(c.orbit.yawDeg).toBe(0);
    el.fire("dblclick", { clientX: 500, clientY: 400, altKey: true });
    expect(c.orbit.pivotZ).toBe(6);
  });

  it("pick mode: a click picks, a drag does not", () => {
    const { el, c } = wired({ depthAt: () => 5 });
    c.togglePickMode();
    el.fire("pointerdown", ptr(1, 300, 300));
    expect(el.style.cursor).toBe("crosshair");
    el.fire("pointermove", ptr(1, 400, 300)); // a drag in pick mode does not orbit
    expect(c.orbit.yawDeg).toBe(0);
    el.fire("pointerup", ptr(1, 400, 300));
    expect(c.orbit.pivotZ).toBe(4); // moved too far: not a click

    expect(c.pickMode).toBe(true); // still on: the drag did not pick
    el.fire("pointerdown", ptr(1, 300, 300));
    el.fire("pointerup", ptr(1, 300, 300));
    expect(c.orbit.pivotZ).toBe(5);
    expect(c.pickMode).toBe(false);
  });

  it("keyboard shortcuts act and prevent default, modifiers pass through", () => {
    const { el, c } = wired();
    const r = el.fire("keydown", { key: "ArrowLeft", shiftKey: true });
    expect(r.prevented).toBe(true);
    expect(c.orbit.yawDeg).toBeCloseTo(5, 9);
    expect(el.fire("keydown", { key: "ArrowLeft", shiftKey: false, ctrlKey: true }).prevented).toBe(
      false,
    );
    expect(el.fire("keydown", { key: "q", shiftKey: false }).prevented).toBe(false);
    el.fire("keydown", { key: "p", shiftKey: false });
    expect(el.style.cursor).toBe("crosshair");
  });

  it("suppresses the context menu so right-drag can pan", () => {
    const { el } = wired();
    expect(el.fire("contextmenu", {}).prevented).toBe(true);
  });

  it("detach removes every listener and later events change nothing", () => {
    const { el, c, detach } = wired();
    expect(el.count).toBeGreaterThan(6);
    detach();
    expect(el.count).toBe(0);
    el.fire("pointerdown", ptr(1, 500, 400));
    el.fire("pointermove", ptr(1, 300, 400));
    el.fire("wheel", { deltaY: -100 });
    expect(c.orbit.yawDeg).toBe(0);
    expect(c.orbit.dolly).toBe(0);
  });
});
