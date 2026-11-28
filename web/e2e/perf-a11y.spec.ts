import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { makeJpeg, openApp, uploadAndWaitReady } from "./helpers";

test("accessibility: no serious or critical violations on the empty state and the viewer", async ({
  page,
}, info) => {
  await openApp(page);
  const serious = (r: Awaited<ReturnType<AxeBuilder["analyze"]>>) =>
    r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  const empty = await new AxeBuilder({ page }).analyze();
  info.annotations.push({
    type: "axe-empty",
    description: JSON.stringify(empty.violations.map((v) => [v.id, v.impact, v.nodes.length])),
  });
  expect(
    serious(empty),
    JSON.stringify(
      serious(empty).map((v) => [v.id, v.help]),
      null,
      1,
    ),
  ).toEqual([]);

  await uploadAndWaitReady(page, await makeJpeg(page, { seed: 31 }));
  await page.waitForTimeout(700); // the viewer fades in over 300 ms
  const viewer = await new AxeBuilder({ page }).analyze();
  info.annotations.push({
    type: "axe-viewer",
    description: JSON.stringify(viewer.violations.map((v) => [v.id, v.impact, v.nodes.length])),
  });
  expect(
    serious(viewer),
    JSON.stringify(
      serious(viewer).map((v) => [v.id, v.help]),
      null,
      1,
    ),
  ).toEqual([]);
});

test("frame rate: orbit for 3 s at 2x mesh on a 1080p viewport (Chromium, hardware GL only)", async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "frame timing is measured on Chromium");
  test.skip(!!process.env.PLAYWRIGHT_REAL, "the stub suite measures the renderer only");
  await page.setViewportSize({ width: 1920, height: 1080 });
  await openApp(page);
  await uploadAndWaitReady(page, await makeJpeg(page, { width: 2048, height: 1536, seed: 32 }));
  const renderer = await page.evaluate(() => {
    const gl = document.createElement("canvas").getContext("webgl2") as WebGL2RenderingContext;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "unknown";
  });
  test.skip(
    /swiftshader|llvmpipe|software/i.test(renderer),
    `software renderer (${renderer}): no frame-rate claim`,
  );

  const stats = await page.evaluate(async () => {
    const nvs = (
      window as unknown as {
        __nvs: { controller: { setOrbit(o: unknown): void; orbit: { pivotZ: number } } };
      }
    ).__nvs;
    const stamps: number[] = [];
    const t0 = performance.now();
    await new Promise<void>((resolve) => {
      const step = (now: number): void => {
        stamps.push(now);
        const s = (now - t0) / 1000;
        nvs.controller.setOrbit({
          yawDeg: 10 * Math.sin(s * 1.3),
          pitchDeg: 4 * Math.sin(s * 0.9),
          dolly: 0,
          pan: [0, 0],
          pivotZ: nvs.controller.orbit.pivotZ,
        });
        if (now - t0 < 3000) requestAnimationFrame(step);
        else resolve();
      };
      requestAnimationFrame(step);
    });
    const dt = stamps
      .slice(1)
      .map((t, i) => t - (stamps[i] as number))
      .sort((a, b) => a - b);
    const pick = (p: number): number =>
      dt[Math.min(dt.length - 1, Math.floor(p * dt.length))] as number;
    return { frames: dt.length, p50: pick(0.5), p95: pick(0.95), max: dt[dt.length - 1] as number };
  });
  console.log(`FRAMES on ${renderer}: ${JSON.stringify(stats)}`);
  expect(stats.frames).toBeGreaterThan(100);
  // 60 Hz displays quantise to 16.7 ms: allow a small jitter on top of the ticket's 16.7 ms
  expect(stats.p95).toBeLessThanOrEqual(17.5);
});
