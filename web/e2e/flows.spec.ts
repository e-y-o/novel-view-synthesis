import { expect, test } from "@playwright/test";
import { canvasFingerprint, makeJpeg, openApp, uploadAndWaitReady } from "./helpers";

test("smoke: upload, viewer, drag changes the pose and the picture", async ({ page }, info) => {
  await openApp(page);
  const jpeg = await makeJpeg(page, { seed: 1 });
  await uploadAndWaitReady(page, jpeg);
  await page.screenshot({ path: info.outputPath("viewer.png") });

  const pose = page.getByTestId("pose");
  await expect(pose).toContainText("yaw +0.0°");
  const before = await canvasFingerprint(page);

  const box = (await page.locator(".viewport-canvas").boundingBox())!;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx - 80, cy, { steps: 8 });
  await page.mouse.up();

  await expect(pose).not.toContainText("yaw +0.0°");
  await expect.poll(() => canvasFingerprint(page)).not.toBe(before);
  // the coverage meter answers for the new view
  await expect(page.getByTestId("meter-total")).toContainText("% will be filled");
});

test("reload on #/s/<id> brings the viewer back; deleting from Recent returns to the drop zone", async ({
  page,
}) => {
  await openApp(page);
  const id = await uploadAndWaitReady(page, await makeJpeg(page, { seed: 2 }), "reload me.jpg");
  await page.reload();
  await expect(page.locator(".viewport-canvas.ready")).toBeVisible();
  await expect(page.locator(".filename")).toHaveText("reload me.jpg");
  expect(await page.evaluate(() => location.hash)).toBe(`#/s/${id}`);

  await page.getByRole("button", { name: "Delete reload me.jpg" }).click();
  await expect(page.getByText("Drop a photo here")).toBeVisible();
  expect(await page.evaluate(() => location.hash)).toBe("#/");
});

test("a text file named .jpg gets a specific message (server side)", async ({ page }) => {
  await openApp(page);
  await page.getByTestId("file-input").setInputFiles({
    name: "notes.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("this is not an image"),
  });
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("This file could not be read");
  await expect(alert).toContainText("JPEG, PNG, WebP or HEIC");
  await expect(alert.getByRole("button", { name: "Choose another photo" })).toBeVisible();
  await alert.getByRole("button", { name: "Dismiss" }).click();
  await expect(alert).toBeHidden();
});

test("an unsupported type is rejected in the browser before any upload", async ({ page }) => {
  await openApp(page);
  let uploads = 0;
  page.on(
    "request",
    (r) => r.url().endsWith("/api/sessions") && r.method() === "POST" && (uploads += 1),
  );
  await page.getByTestId("file-input").setInputFiles({
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("hello"),
  });
  await expect(page.getByRole("alert")).toContainText("notes.txt");
  expect(uploads).toBe(0);
});

test("worker killed: the render fails with a retry button, and retry recovers", async ({
  page,
  request,
}) => {
  await openApp(page);
  await uploadAndWaitReady(page, await makeJpeg(page, { seed: 3 }));

  const killed = await request.post("/api/debug/kill-worker?restart=false");
  expect((await killed.json()).killed).toBe(true);
  await expect(page.locator(".dot.error, .dot.starting").first()).toBeVisible({ timeout: 15_000 });

  await page.getByRole("button", { name: /Render full resolution/ }).click();
  const inline = page.locator(".inline-error");
  await expect(inline).toBeVisible({ timeout: 15_000 });
  await expect(inline.getByRole("button", { name: "Retry" })).toBeVisible();

  await request.post("/api/debug/restart-worker");
  await expect(page.locator(".dot.ready")).toBeVisible({ timeout: 30_000 });
  await inline.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByTestId("render-result")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("render-info")).toContainText("640 × 480 px");
});

test("render: result overlay, outdated badge after moving, download is a PNG of the source size", async ({
  page,
}) => {
  await openApp(page);
  await uploadAndWaitReady(page, await makeJpeg(page, { seed: 4 }), "my photo.jpg");

  // move a little, then render
  await page.evaluate(() => {
    const nvs = (
      window as unknown as { __nvs: { controller: { setOrbit(o: unknown): void; orbit: object } } }
    ).__nvs;
    nvs.controller.setOrbit({ ...nvs.controller.orbit, yawDeg: 4 });
  });
  await page.getByRole("button", { name: /Render full resolution/ }).click();
  await expect(page.getByTestId("render-result")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".result-overlay img")).toBeVisible();
  await expect(page.getByTestId("stale-badge")).toHaveCount(0);

  // Space held compares with the preview without changing the toggle: switch to Preview first
  await page.getByRole("button", { name: "Preview" }).click();
  await expect(page.locator(".result-overlay")).toHaveCount(0);
  await page.locator(".viewport-canvas").focus();
  await page.keyboard.down(" ");
  await expect(page.locator(".result-overlay img")).toBeVisible();
  await page.keyboard.up(" ");
  await expect(page.locator(".result-overlay")).toHaveCount(0);

  // moving the camera marks the result outdated but keeps it
  await page.evaluate(() => {
    const nvs = (
      window as unknown as { __nvs: { controller: { setOrbit(o: unknown): void; orbit: object } } }
    ).__nvs;
    nvs.controller.setOrbit({ ...nvs.controller.orbit, yawDeg: 9 });
  });
  await expect(page.getByTestId("stale-badge")).toBeVisible();

  const link = page.getByTestId("download");
  const href = (await link.getAttribute("href"))!;
  const response = await page.request.get(href);
  expect(response.status()).toBe(200);
  expect(response.headers()["content-type"]).toBe("image/png");
  expect(response.headers()["content-disposition"]).toBe(
    'attachment; filename="my photo-view-yaw+4.0-pitch+0.0.png"',
  );
  const body = await response.body();
  expect(body.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
  expect(body.readUInt32BE(16)).toBe(640); // IHDR width
  expect(body.readUInt32BE(20)).toBe(480); // IHDR height
});
