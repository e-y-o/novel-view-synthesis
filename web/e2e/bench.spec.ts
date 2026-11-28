import { expect, test } from "@playwright/test";
import { makeJpeg, openApp } from "./helpers";

/**
 * Opt-in browser-side benchmark (`PLAYWRIGHT_REAL=1 PLAYWRIGHT_BENCH=1 npm run e2e:chromium -- bench`): real
 * model, generated photos. Reports (1) upload to first rendered frame for 2 / 12 / 24 MP, cold and warm, and (2) a
 * 10 s synthetic drag with the coverage meter running: frame intervals from requestAnimationFrame, coverage round
 * trips seen by the page. Do not run it while another GPU job (the server benchmark) is running.
 */
test.skip(
  !process.env.PLAYWRIGHT_BENCH,
  "set PLAYWRIGHT_BENCH=1 (and PLAYWRIGHT_REAL=1) to run the benchmark",
);

const SIZES: [string, number, number][] = [
  ["2 MP", 1600, 1250],
  ["12 MP", 4000, 3000],
  ["24 MP", 6000, 4000],
];
const median = (v: number[]): number =>
  [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] as number;

test("upload to first frame, cold and warm", async ({ page }) => {
  test.setTimeout(600_000);
  await openApp(page);
  await expect(page.locator(".dot.ready")).toBeVisible({ timeout: 120_000 });
  const table: Record<string, { cold_ms: number; warm_ms: number }> = {};
  for (const [label, w, h] of SIZES) {
    const cold: number[] = [];
    const warm: number[] = [];
    for (let i = 0; i < 3; i += 1) {
      const jpeg = await makeJpeg(page, {
        width: w,
        height: h,
        texture: true,
        seed: 700 + i * 10 + w,
      });
      for (const kind of ["cold", "warm"] as const) {
        await page.goto("/?e2e=1#/");
        await expect(page.getByRole("group", { name: "Add a photo" })).toBeVisible();
        const t0 = Date.now();
        await page
          .getByTestId("file-input")
          .setInputFiles({ name: `${label}.jpg`, mimeType: "image/jpeg", buffer: jpeg });
        await expect(page.locator(".viewport-canvas.ready")).toBeVisible({ timeout: 90_000 });
        await page.evaluate(
          () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))),
        ); // first frame drawn
        (kind === "cold" ? cold : warm).push(Date.now() - t0);
      }
    }
    table[label] = { cold_ms: median(cold), warm_ms: median(warm) };
    console.log(
      `BENCH first frame ${label}: ${JSON.stringify(table[label])} (cold ${cold}, warm ${warm})`,
    );
  }
});

test("10 s drag with the coverage meter running", async ({ page }) => {
  test.setTimeout(120_000);
  await openApp(page);
  await expect(page.locator(".dot.ready")).toBeVisible({ timeout: 120_000 });
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.getByTestId("file-input").setInputFiles({
    name: "drag.jpg",
    mimeType: "image/jpeg",
    buffer: await makeJpeg(page, { width: 4000, height: 3000, texture: true, seed: 999 }),
  });
  await expect(page.locator(".viewport-canvas.ready")).toBeVisible({ timeout: 90_000 });
  await page.waitForTimeout(1500);

  const stats = await page.evaluate(async () => {
    const coverage: number[] = [];
    const orig = window.fetch;
    window.fetch = async (...args) => {
      const isCov = String(args[0]).endsWith("/coverage");
      const t0 = performance.now();
      try {
        return await orig(...args);
      } finally {
        if (isCov) coverage.push(performance.now() - t0);
      }
    };
    const el = document.querySelector(".viewport-canvas") as HTMLElement;
    const r = el.getBoundingClientRect();
    const mk = (type: string, x: number, y: number): PointerEvent =>
      new PointerEvent(type, {
        pointerId: 5,
        clientX: x,
        clientY: y,
        button: 0,
        buttons: 1,
        bubbles: true,
        cancelable: true,
      });
    const frames: number[] = [];
    let running = true;
    const raf = (now: number): void => {
      frames.push(now);
      if (running) requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
    const t0 = performance.now();
    el.dispatchEvent(mk("pointerdown", r.left + r.width / 2, r.top + r.height / 2));
    while (performance.now() - t0 < 10_000) {
      const s = (performance.now() - t0) / 1000;
      el.dispatchEvent(
        mk(
          "pointermove",
          r.left + r.width / 2 + 220 * Math.sin(s * 1.7),
          r.top + r.height / 2 + 60 * Math.sin(s * 1.1),
        ),
      );
      await new Promise((res) => setTimeout(res, 8));
    }
    el.dispatchEvent(mk("pointerup", r.left + r.width / 2, r.top + r.height / 2));
    running = false;
    const dt = frames
      .slice(1)
      .map((t, i) => t - (frames[i] as number))
      .sort((a, b) => a - b);
    const cov = [...coverage].sort((a, b) => a - b);
    const pick = (a: number[], p: number): number =>
      a[Math.min(a.length - 1, Math.floor(p * a.length))] as number;
    return {
      frames: dt.length,
      fps: dt.length / 10,
      frame_ms_p50: pick(dt, 0.5),
      frame_ms_p95: pick(dt, 0.95),
      coverage_requests: cov.length,
      coverage_per_s: cov.length / 10,
      coverage_ms_p50: pick(cov, 0.5),
      coverage_ms_p95: pick(cov, 0.95),
    };
  });
  console.log(`BENCH drag: ${JSON.stringify(stats)}`);
  expect(stats.frames).toBeGreaterThan(100);
});
