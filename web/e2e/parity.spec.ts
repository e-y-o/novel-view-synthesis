/**
 * Browser vs server parity (stub depth, deterministic): the WebGL preview must show the same view as the server's
 * splat of the same pose (`POST /render` with `size: "preview"`, holes magenta). The canvas is set to the preview
 * size (scale 1, no letterbox) and compared on pixels valid in both renderers, eroded by 3 px.
 *
 * Measured on Chromium (Chrome channel, ANGLE/D3D11 on an RTX 5070 Ti; 640x480 generated scenes, yaw +-4 and +-8,
 * pitch 4; gradient scene and textured scene):
 *   MAE over pixels valid in both (0..255): gradient 0.02 to 0.05, textured 0.20 to 0.35 (worst 0.351)
 *   yellow marker centroid distance: worst 0.014 px (marker about 23 800 pixels in both pictures)
 *   share of the view valid in both: 0.77 (yaw 8) to 0.91 (pitch 4)
 * Thresholds are 1.5 x the worst Chromium value, rounded up (the ticket asks for the safety factor): MAE 0.55,
 * centroid 0.03 px. Other engines use the ticket's looser 1.5 px and a MAE of 3 until measured there.
 */

import { expect, test } from "@playwright/test";
import { makeJpeg, openApp, uploadAndWaitReady } from "./helpers";

const CHROMIUM = { mae: 0.55, centroid: 0.03 };
const OTHER_ENGINES = { mae: 3, centroid: 1.5 }; // ticket: centroid <= 1.5 px
const MIN_BOTH_FRACTION = 0.3; // both renderers must agree that a good part of the view is real

interface Pose {
  yaw: number;
  pitch: number;
}
const POSES: Pose[] = [
  { yaw: 4, pitch: 0 },
  { yaw: -4, pitch: 0 },
  { yaw: 8, pitch: 0 },
  { yaw: -8, pitch: 0 },
  { yaw: 0, pitch: 4 },
];

interface Metrics {
  mae: number;
  bothFraction: number;
  browserCentroid: [number, number] | null;
  serverCentroid: [number, number] | null;
  centroidDistance: number | null;
  markerPixels: [number, number];
}

test.skip(!!process.env.PLAYWRIGHT_REAL, "parity needs the deterministic stub depth");

for (const scene of [
  { name: "gradient", opts: { seed: 21 } },
  { name: "textured", opts: { seed: 22, texture: true } },
]) {
  test.describe(`parity: ${scene.name} scene`, () => {
    for (const pose of POSES) {
      test(`yaw ${pose.yaw}, pitch ${pose.pitch}`, async ({ page }, info) => {
        await openApp(page);
        await uploadAndWaitReady(page, await makeJpeg(page, scene.opts), `${scene.name}.jpg`);
        const m = await page.evaluate(async (p): Promise<Metrics> => {
          type Nvs = {
            renderer: {
              setFixedSize(s: [number, number] | null): void;
              setOptions(o: object): void;
              setCamera(c: unknown): void;
              render(): void;
            };
            controller: {
              setOrbit(o: unknown): void;
              orbit: { pivotZ: number };
              camera: { R: number[][]; t: number[] };
            };
            canvas: HTMLCanvasElement;
          };
          const nvs = (window as unknown as { __nvs: Nvs }).__nvs;
          const id = location.hash.replace("#/s/", "");
          const meta = await (await fetch(`/api/sessions/${id}/assets/meta.json`)).json();
          const W: number = meta.preview_width;
          const H: number = meta.preview_height;
          nvs.renderer.setFixedSize([W, H]); // canvas = preview size, scale 1, no letterbox
          nvs.controller.setOrbit({
            yawDeg: p.yaw,
            pitchDeg: p.pitch,
            dolly: 0,
            pan: [0, 0],
            pivotZ: nvs.controller.orbit.pivotZ,
          });
          const cam = nvs.controller.camera;
          nvs.renderer.setCamera(cam);

          const gl = nvs.canvas.getContext("webgl2") as WebGL2RenderingContext;
          const read = (): Uint8Array => {
            nvs.renderer.render();
            const raw = new Uint8Array(W * H * 4);
            gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, raw);
            const out = new Uint8Array(W * H * 4);
            for (let y = 0; y < H; y += 1)
              out.set(raw.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
            return out;
          };
          nvs.renderer.setOptions({ background: [0, 0, 0] });
          const onBlack = read();
          nvs.renderer.setOptions({ background: [1, 1, 1] });
          const onWhite = read();
          nvs.renderer.setOptions({ background: [0.14, 0.14, 0.14] });

          // the server's view of the same pose, holes magenta
          const body = {
            R: cam.R,
            t: cam.t,
            size: "preview",
            hole: "magenta",
            fill_specks: false,
            format: "png",
          };
          const res = await (
            await fetch(`/api/sessions/${id}/render`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body),
            })
          ).json();
          const bitmap = await createImageBitmap(await (await fetch(res.url)).blob());
          const oc = new OffscreenCanvas(W, H);
          const ctx = oc.getContext("2d") as OffscreenCanvasRenderingContext2D;
          ctx.drawImage(bitmap, 0, 0);
          const server = ctx.getImageData(0, 0, W, H).data;

          // valid in both
          const valid = new Uint8Array(W * H);
          for (let i = 0; i < W * H; i += 1) {
            const browserFilled =
              Math.abs((onBlack[i * 4] as number) - (onWhite[i * 4] as number)) <= 2 &&
              Math.abs((onBlack[i * 4 + 1] as number) - (onWhite[i * 4 + 1] as number)) <= 2 &&
              Math.abs((onBlack[i * 4 + 2] as number) - (onWhite[i * 4 + 2] as number)) <= 2;
            const magenta =
              (server[i * 4] as number) > 250 &&
              (server[i * 4 + 1] as number) < 5 &&
              (server[i * 4 + 2] as number) > 250;
            valid[i] = browserFilled && !magenta ? 1 : 0;
          }
          // erode by 3 px (chebyshev) so cut edges and splat footprints do not count
          const R = 3;
          const eroded = new Uint8Array(W * H);
          for (let y = R; y < H - R; y += 1) {
            for (let x = R; x < W - R; x += 1) {
              let ok = 1;
              for (let dy = -R; dy <= R && ok; dy += 1)
                for (let dx = -R; dx <= R; dx += 1)
                  if (!valid[(y + dy) * W + (x + dx)]) {
                    ok = 0;
                    break;
                  }
              eroded[y * W + x] = ok;
            }
          }
          let sum = 0;
          let n = 0;
          for (let i = 0; i < W * H; i += 1) {
            if (!eroded[i]) continue;
            sum +=
              (Math.abs((onBlack[i * 4] as number) - (server[i * 4] as number)) +
                Math.abs((onBlack[i * 4 + 1] as number) - (server[i * 4 + 1] as number)) +
                Math.abs((onBlack[i * 4 + 2] as number) - (server[i * 4 + 2] as number))) /
              3;
            n += 1;
          }

          // the yellow marker disc: centroid in both pictures
          const centroid = (
            px: ArrayLike<number>,
            only: Uint8Array | null,
          ): [[number, number] | null, number] => {
            let cx = 0;
            let cy = 0;
            let c = 0;
            for (let i = 0; i < W * H; i += 1) {
              if (only && !only[i]) continue;
              const r = px[i * 4] as number;
              const g = px[i * 4 + 1] as number;
              const b = px[i * 4 + 2] as number;
              if (r > 200 && g > 170 && b < 90) {
                cx += i % W;
                cy += Math.floor(i / W);
                c += 1;
              }
            }
            return [c ? [cx / c, cy / c] : null, c];
          };
          const [bc, bn] = centroid(onBlack, null);
          const [sc, sn] = centroid(server, null);
          nvs.renderer.setFixedSize(null);
          return {
            mae: n ? sum / n : NaN,
            bothFraction: n / (W * H),
            browserCentroid: bc,
            serverCentroid: sc,
            centroidDistance: bc && sc ? Math.hypot(bc[0] - sc[0], bc[1] - sc[1]) : null,
            markerPixels: [bn, sn],
          };
        }, pose);

        info.annotations.push({
          type: "parity",
          description: JSON.stringify({ scene: scene.name, ...pose, ...m }),
        });
        console.log(
          `PARITY ${scene.name} yaw ${pose.yaw} pitch ${pose.pitch}: ${JSON.stringify(m)}`,
        );
        expect(m.markerPixels[0]).toBeGreaterThan(200);
        expect(m.markerPixels[1]).toBeGreaterThan(200);
        expect(m.bothFraction).toBeGreaterThan(MIN_BOTH_FRACTION);
        const limit = info.project.name === "chromium" ? CHROMIUM : OTHER_ENGINES;
        expect(m.mae).toBeLessThanOrEqual(limit.mae);
        expect(m.centroidDistance).not.toBeNull();
        expect(m.centroidDistance as number).toBeLessThanOrEqual(limit.centroid);
      });
    }
  });
}
