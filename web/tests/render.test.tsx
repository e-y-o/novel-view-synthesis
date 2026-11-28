// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../src/api/client";
import {
  buildRenderBody,
  downloadName,
  isStale,
  poseKey,
  safeStem,
  type RenderOptions,
  type RenderResult,
} from "../src/api/render";
import { defaultOrbit, orbitPose, type CameraController, type OrbitParams } from "../src/camera";
import {
  cancelRender,
  renderOptions,
  renderState,
  resetRender,
  showResult,
  startRender,
  type RenderDeps,
} from "../src/state/render";
import { RenderPanel } from "../src/ui/RenderPanel";
import { ResultOverlay } from "../src/ui/ResultOverlay";

afterEach(cleanup);

const OPTS: RenderOptions = { hole: "transparent", fillSpecks: true };
const ORBIT: OrbitParams = {
  yawDeg: 5.123456789,
  pitchDeg: -2.5,
  dolly: 0.31,
  pan: [0.1, -0.2],
  pivotZ: 3.2,
};

describe("request body", () => {
  it("carries the pose rounded to 1e-6, the orbit, source size and PNG", () => {
    const body = buildRenderBody(orbitPose(ORBIT), ORBIT, OPTS);
    expect(Object.keys(body).sort()).toEqual([
      "R",
      "fill_specks",
      "format",
      "hole",
      "orbit",
      "size",
      "t",
    ]);
    expect(body.size).toBe("source");
    expect(body.format).toBe("png");
    expect(body.fill_specks).toBe(true);
    expect(body.hole).toBe("alpha"); // the UI says transparent, the API says alpha
    expect(body.orbit).toEqual({
      yaw_deg: 5.123456789,
      pitch_deg: -2.5,
      dolly: 0.31,
      pan: [0.1, -0.2],
      pivot_z: 3.2,
    });
    for (const v of [...body.R.flat(), ...body.t]) {
      expect(Math.abs(v * 1e6 - Math.round(v * 1e6))).toBeLessThan(1e-6 * 10); // at most 6 decimals
    }
    expect(body.R.map((r) => r.length)).toEqual([3, 3, 3]);
    const exact = orbitPose(ORBIT);
    expect(Math.abs(body.t[0]! - exact.t[0])).toBeLessThan(5e-7);
  });

  it("maps the hole options and never sends -0", () => {
    expect(
      buildRenderBody(orbitPose(defaultOrbit(3)), defaultOrbit(3), { ...OPTS, hole: "magenta" })
        .hole,
    ).toBe("magenta");
    expect(
      buildRenderBody(orbitPose(defaultOrbit(3)), defaultOrbit(3), { ...OPTS, hole: "black" }).hole,
    ).toBe("black");
    const body = buildRenderBody(orbitPose(defaultOrbit(3)), defaultOrbit(3), {
      ...OPTS,
      fillSpecks: false,
    });
    expect(body.fill_specks).toBe(false);
    for (const v of [...body.R.flat(), ...body.t]) expect(Object.is(v, -0)).toBe(false);
  });
});

describe("stale logic", () => {
  it("is outdated only when the view or an option that changes pixels changed", () => {
    const key = poseKey(ORBIT, OPTS);
    expect(isStale(key, poseKey(ORBIT, OPTS))).toBe(false);
    expect(isStale(key, poseKey({ ...ORBIT, yawDeg: ORBIT.yawDeg + 1e-6 }, OPTS))).toBe(false); // below 1e-4
    expect(isStale(key, poseKey({ ...ORBIT, yawDeg: ORBIT.yawDeg + 0.01 }, OPTS))).toBe(true);
    expect(isStale(key, poseKey({ ...ORBIT, dolly: 0.5 }, OPTS))).toBe(true);
    expect(isStale(key, poseKey({ ...ORBIT, pan: [0.1, 0] }, OPTS))).toBe(true);
    expect(isStale(key, poseKey(ORBIT, { ...OPTS, hole: "black" }))).toBe(true);
    expect(isStale(key, poseKey(ORBIT, { ...OPTS, fillSpecks: false }))).toBe(true);
    expect(isStale(null, key)).toBe(false);
    expect(isStale(key, null)).toBe(false);
  });
});

describe("download name", () => {
  it("is <stem>-view-yaw{+5.0}-pitch{+0.0}.png with signed one-decimal angles", () => {
    expect(downloadName("holiday.jpg", { ...ORBIT, yawDeg: 5, pitchDeg: 0 })).toBe(
      "holiday-view-yaw+5.0-pitch+0.0.png",
    );
    expect(downloadName("holiday.jpg", { ...ORBIT, yawDeg: -12.34, pitchDeg: 3.06 })).toBe(
      "holiday-view-yaw-12.3-pitch+3.1.png",
    );
    expect(downloadName("a.jpg", { ...ORBIT, yawDeg: -0.01, pitchDeg: -0 })).toBe(
      "a-view-yaw+0.0-pitch+0.0.png",
    );
  });

  it("is ASCII only and Windows-safe", () => {
    for (const [input, stem] of [
      ["café déjà vu.heic", "cafe deja vu"],
      ["日本語の写真.jpg", "image"],
      ['a<b>c:d"e/f\\g|h?i*j.png', "a_b_c_d_e_f_g_h_i_j"],
      ["CON.jpg", "_CON"],
      ["lpt1", "_lpt1"],
      ["  ..hidden..  .jpg", "hidden"],
      ["trailing dots...jpg", "trailing dots"],
      ["", "image"],
      ["IMG_0001.JPEG", "IMG_0001"],
      ["my.photo.v2.jpg", "my.photo.v2"],
      ["emoji 🙂 face.png", "emoji _ face"],
    ] as const) {
      expect(safeStem(input), input).toBe(stem);
    }
    const name = downloadName("名前 é<>.jpg", ORBIT);
    expect(/^[\x20-\x7e]+$/.test(name)).toBe(true);
    expect(name).not.toMatch(/[<>:"/\\|?*]/);
  });

  it("caps very long names", () => {
    const name = downloadName("x".repeat(400) + ".jpg", ORBIT);
    expect(name.length).toBeLessThan(100);
    expect(name.endsWith("-pitch-2.5.png")).toBe(true);
  });
});

// ------------------------------------------------------------------------------------- single flight
const RESULT: RenderResult = {
  url: "/api/sessions/s/results/abc.png",
  name: "abc.png",
  width: 4032,
  height: 3024,
  coverage: { disocclusion: 0.05, stretch: 0.01, out_of_frame: 0.02, total: 0.08 },
  ms: { load: 3, splat: 90, classify: 16, compose: 0.3, save: 400, total: 510 },
};

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("startRender", () => {
  beforeEach(() => {
    resetRender();
    renderOptions.value = { hole: "transparent", fillSpecks: true };
  });
  afterEach(() => resetRender());

  const cam = orbitPose(ORBIT);

  it("runs one render at a time: a second call while one is running does nothing", async () => {
    const d = deferred<RenderResult>();
    const send = vi.fn((_id: string, _body: unknown, _signal: AbortSignal) => d.promise);
    const deps: RenderDeps = { send, now: () => 0 };
    const first = startRender("s", cam, ORBIT, "p.jpg", deps);
    expect(renderState.value.status).toBe("running");
    expect(await startRender("s", cam, ORBIT, "p.jpg", deps)).toBe(false);
    expect(await startRender("s", cam, ORBIT, "p.jpg", deps)).toBe(false);
    expect(send).toHaveBeenCalledOnce();
    d.resolve(RESULT);
    await first;
    expect(renderState.value.status).toBe("done");
    expect(renderState.value.outcome?.result).toEqual(RESULT);
    expect(renderState.value.outcome?.key).toBe(poseKey(ORBIT, OPTS));
    expect(showResult.value).toBe(true);
    // and another can start afterwards
    send.mockResolvedValueOnce(RESULT);
    expect(await startRender("s", cam, ORBIT, "p.jpg", deps)).toBe(true);
    expect(send).toHaveBeenCalledTimes(2);
  });

  it("sends the options chosen in the panel", async () => {
    renderOptions.value = { hole: "magenta", fillSpecks: false };
    const send = vi.fn(async () => RESULT);
    await startRender("sess", cam, ORBIT, "p.jpg", { send, now: () => 0 });
    const [id, body] = send.mock.calls[0] as unknown as [
      string,
      { hole: string; fill_specks: boolean },
    ];
    expect(id).toBe("sess");
    expect(body.hole).toBe("magenta");
    expect(body.fill_specks).toBe(false);
  });

  it("cancel aborts the request and goes back to idle; an old result stays", async () => {
    const d = deferred<RenderResult>();
    let signal!: AbortSignal;
    const deps: RenderDeps = {
      send: (_id, _b, s) => {
        signal = s;
        return d.promise;
      },
      now: () => 0,
    };
    const run = startRender("s", cam, ORBIT, "p.jpg", deps);
    cancelRender();
    expect(signal.aborted).toBe(true);
    expect(renderState.value.status).toBe("idle");
    d.reject(new DOMException("aborted", "AbortError"));
    await run;
    expect(renderState.value.status).toBe("idle");
    expect(renderState.value.error).toBeNull();

    // with an earlier result, cancelling a re-render keeps showing it
    await startRender("s", cam, ORBIT, "p.jpg", { send: async () => RESULT, now: () => 0 });
    const d2 = deferred<RenderResult>();
    const again = startRender("s", cam, ORBIT, "p.jpg", { send: () => d2.promise, now: () => 0 });
    cancelRender();
    d2.reject(new DOMException("aborted", "AbortError"));
    await again;
    expect(renderState.value.status).toBe("done");
    expect(renderState.value.outcome?.result).toEqual(RESULT);
  });

  it("maps server errors to a specific message with a retry, and keeps the old result", async () => {
    await startRender("s", cam, ORBIT, "p.jpg", { send: async () => RESULT, now: () => 0 });
    const fail = async () => {
      throw new ApiError(507, "gpu_oom", "out of memory");
    };
    await startRender("s", cam, ORBIT, "p.jpg", { send: fail, now: () => 0 });
    const s = renderState.value;
    expect(s.status).toBe("error");
    expect(s.error?.title).toBe("Not enough GPU memory");
    expect(s.error?.action).toBe("retry");
    expect(s.outcome?.result).toEqual(RESULT);
    for (const [code, status, title] of [
      ["not_ready", 409, /Something went wrong|ready/i],
      ["timeout", 504, /too long/],
      ["worker_died", 503, /worker/i],
    ] as const) {
      await startRender("s", cam, ORBIT, "p.jpg", {
        send: async () => {
          throw new ApiError(status, code, "x");
        },
        now: () => 0,
      });
      expect(`${renderState.value.error?.title}`).toMatch(title);
    }
  });
});

// ---------------------------------------------------------------------------------------- the panel
describe("RenderPanel and overlay", () => {
  beforeEach(() => {
    resetRender();
    renderOptions.value = { hole: "transparent", fillSpecks: true };
  });
  afterEach(() => resetRender());

  const controller = {
    camera: orbitPose(ORBIT),
    orbit: ORBIT,
  } as unknown as CameraController;

  it("is disabled until a viewer exists and then renders the controller's view", () => {
    const { rerender } = render(
      <RenderPanel sessionId={null} filename="p.jpg" controller={null} pose={null} />,
    );
    expect(
      (screen.getByRole("button", { name: /Render full resolution/ }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    rerender(<RenderPanel sessionId="s" filename="p.jpg" controller={controller} pose={ORBIT} />);
    expect(
      (screen.getByRole("button", { name: /Render full resolution/ }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
    expect(screen.getByText(/Holes are marked, not filled/)).toBeTruthy();
  });

  it("shows the result details, a download link with the built name, and marks it outdated", async () => {
    await startRender("s", orbitPose(ORBIT), ORBIT, "My Photo.jpg", {
      send: async () => RESULT,
      now: () => 0,
    });
    const { rerender } = render(
      <RenderPanel sessionId="s" filename="My Photo.jpg" controller={controller} pose={ORBIT} />,
    );
    expect(screen.getByTestId("render-info").textContent).toContain("4032 × 3024 px");
    expect(screen.getByTestId("render-info").textContent).toContain("8.0 % holes");
    expect(screen.getByTestId("render-info").textContent).toContain("510 ms");
    const link = screen.getByTestId("download") as HTMLAnchorElement;
    const name = "My Photo-view-yaw+5.1-pitch-2.5.png";
    expect(link.getAttribute("download")).toBe(name);
    expect(link.getAttribute("href")).toBe(
      `${RESULT.url}?download=1&filename=${encodeURIComponent(name)}`,
    );
    expect(screen.queryByTestId("stale-badge")).toBeNull();
    rerender(
      <RenderPanel
        sessionId="s"
        filename="My Photo.jpg"
        controller={controller}
        pose={{ ...ORBIT, yawDeg: 9 }}
      />,
    );
    expect(screen.getByTestId("stale-badge").textContent).toContain("outdated");
    expect(screen.getByRole("button", { name: "Render again" })).toBeTruthy();
  });

  it("toggles between preview and result, and the overlay never takes focus or the pointer", async () => {
    await startRender("s", orbitPose(ORBIT), ORBIT, "p.jpg", {
      send: async () => RESULT,
      now: () => 0,
    });
    showResult.value = false;
    const view = render(
      <div>
        <RenderPanel sessionId="s" filename="p.jpg" controller={controller} pose={ORBIT} />
        <ResultOverlay />
      </div>,
    );
    expect(view.container.querySelector(".result-overlay")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Result" }));
    view.rerender(
      <div>
        <RenderPanel sessionId="s" filename="p.jpg" controller={controller} pose={ORBIT} />
        <ResultOverlay />
      </div>,
    );
    const overlay = view.container.querySelector(".result-overlay");
    expect(overlay).not.toBeNull();
    expect(overlay?.classList.contains("checker")).toBe(true); // transparent holes on a checkerboard
    expect(overlay?.querySelectorAll("a, button, input, [tabindex]").length).toBe(0); // nothing focusable
    expect(screen.getByRole("img", { name: /4032 by 3024/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Result" }).getAttribute("aria-pressed")).toBe(
      "true",
    );
  });

  it("shows progress with a cancel button while rendering", () => {
    renderState.value = { status: "running", startedAt: 0, outcome: null, error: null };
    render(<RenderPanel sessionId="s" filename="p.jpg" controller={controller} pose={ORBIT} />);
    expect(screen.getByTestId("render-elapsed").textContent).toContain("Rendering");
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Render full resolution/ })).toBeNull();
  });
});
