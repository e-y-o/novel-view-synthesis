import { expect, type Page } from "@playwright/test";

/** Open the app with the test hook enabled (`window.__nvs` once a viewer exists). */
export async function openApp(page: Page, hash = ""): Promise<void> {
  await page.goto(`/?e2e=1${hash}`);
  await expect(page.getByRole("button", { name: /back to start/ })).toBeVisible();
}

/**
 * A generated JPEG (never the developer's photos): a vertical gradient with a bright yellow disc marker at the
 * centre (`marker`) and optional high-frequency texture. Created in the page with a canvas.
 */
export async function makeJpeg(
  page: Page,
  opts: {
    width?: number;
    height?: number;
    texture?: boolean;
    marker?: boolean;
    seed?: number;
  } = {},
): Promise<Buffer> {
  const { width = 640, height = 480, texture = false, marker = true, seed = 0 } = opts;
  const base64 = await page.evaluate(
    async ({ width, height, texture, marker, seed }) => {
      const c = document.createElement("canvas");
      c.width = width;
      c.height = height;
      const x = c.getContext("2d") as CanvasRenderingContext2D;
      const g = x.createLinearGradient(0, 0, 0, height);
      g.addColorStop(0, "#3a6fb5");
      g.addColorStop(0.5, "#8fb4d9");
      g.addColorStop(1, "#2f6b3a");
      x.fillStyle = g;
      x.fillRect(0, 0, width, height);
      if (texture) {
        for (let i = 0; i < 1500; i += 1) {
          x.fillStyle = `hsl(${(i * 47) % 360} 40% ${30 + ((i * 13) % 40)}%)`;
          x.fillRect((i * 97) % width, (i * 53) % height, 6, 6);
        }
      }
      // a few seeded specks: different tests get different bytes (sessions are keyed by content)
      for (let i = 0; i < 30; i += 1) {
        x.fillStyle = `rgb(${(seed * 37 + i * 11) % 200} 60 90)`;
        x.fillRect((seed * 131 + i * 71) % (width - 4), (seed * 17 + i * 43) % (height - 4), 3, 3);
      }
      if (marker) {
        x.fillStyle = "rgb(255, 230, 0)"; // pure yellow: nothing else in the picture is near it
        x.beginPath();
        x.ellipse(width / 2, height / 2, width * 0.14, height * 0.18, 0, 0, Math.PI * 2);
        x.fill();
      }
      const blob: Blob = await new Promise((r) =>
        c.toBlob((b) => r(b as Blob), "image/jpeg", 0.92),
      );
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let s = "";
      for (const b of bytes) s += String.fromCharCode(b);
      return btoa(s);
    },
    { width, height, texture, marker, seed },
  );
  return Buffer.from(base64, "base64");
}

/** Upload through the file input and wait for the viewer. Returns the session id from the URL hash. */
export async function uploadAndWaitReady(
  page: Page,
  buffer: Buffer,
  name = "scene.jpg",
): Promise<string> {
  await page.getByTestId("file-input").setInputFiles({ name, mimeType: "image/jpeg", buffer });
  await expect(page.locator(".viewport-canvas.ready")).toBeVisible({ timeout: 30_000 });
  await page.waitForFunction(() => Boolean((window as unknown as { __nvs?: unknown }).__nvs));
  const hash = await page.evaluate(() => location.hash);
  const id = hash.replace("#/s/", "");
  expect(id).toMatch(/^[0-9a-f]{64}$/);
  return id;
}

/** A hash of the canvas pixels (read right after a draw), to tell whether the picture changed. */
export async function canvasFingerprint(page: Page): Promise<string> {
  return page.evaluate(() => {
    const nvs = (
      window as unknown as { __nvs: { renderer: { render(): void }; canvas: HTMLCanvasElement } }
    ).__nvs;
    nvs.renderer.render();
    const gl = nvs.canvas.getContext("webgl2") as WebGL2RenderingContext;
    const w = nvs.canvas.width;
    const h = nvs.canvas.height;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let a = 2166136261;
    for (let i = 0; i < px.length; i += 7) a = Math.imul(a ^ (px[i] as number), 16777619);
    return `${w}x${h}:${(a >>> 0).toString(16)}`;
  });
}
