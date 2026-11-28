import { expect, test } from "@playwright/test";
import { makeJpeg, openApp } from "./helpers";

/**
 * Opt-in (`PLAYWRIGHT_REAL=1 npm run e2e:chromium -- real`): the real Depth Pro 2 model and a generated 12 MP photo
 * (no developer photos). Needs a GPU, the model weights and an accepted licence; the server is started with
 * `--accept-license` by the config.
 */
test.skip(!process.env.PLAYWRIGHT_REAL, "set PLAYWRIGHT_REAL=1 to run against the real model");

test("12 MP upload reaches the viewer in under 3.5 s with a warm model", async ({ page }, info) => {
  test.setTimeout(180_000);
  await openApp(page);
  await expect(page.locator(".dot.ready")).toBeVisible({ timeout: 120_000 }); // model loaded
  const jpeg = await makeJpeg(page, { width: 4000, height: 3000, texture: true, seed: 41 });

  // first upload warms CUDA kernels and the allocator; the second (new bytes) is the measurement
  const upload = async (buffer: Buffer, name: string): Promise<number> => {
    await page.goto("/?e2e=1#/");
    const t0 = Date.now();
    await page.getByTestId("file-input").setInputFiles({ name, mimeType: "image/jpeg", buffer });
    await expect(page.locator(".viewport-canvas.ready")).toBeVisible({ timeout: 60_000 });
    return Date.now() - t0;
  };
  const cold = await upload(jpeg, "warmup.jpg");
  const second = await makeJpeg(page, { width: 4000, height: 3000, texture: true, seed: 42 });
  const warm = await upload(second, "measure.jpg");
  console.log(`REAL 12 MP upload to viewer: first ${cold} ms, warm ${warm} ms`);
  await page.waitForTimeout(500);
  await page.screenshot({ path: info.outputPath("real-viewer.png") });
  expect(warm).toBeLessThan(3500);
});
