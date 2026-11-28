import { afterEach, describe, expect, it, vi } from "vitest";
import { floatToHalf, project, unproject } from "../src/camera";
import {
  canvasToImage,
  clipScale,
  containFit,
  fetchAssets,
  gridTexel,
  imageToCanvas,
  insideImage,
  meshRows,
  quadCutMask,
} from "../src/gl";

describe("containFit", () => {
  it("letterboxes a wide canvas around a tall image (pillarbox)", () => {
    const fit = containFit(1920, 1080, 1000, 1500);
    expect(fit.scale).toBeCloseTo(0.72, 12);
    expect(fit.height).toBeCloseTo(1080, 9);
    expect(fit.width).toBeCloseTo(720, 9);
    expect(fit.offsetX).toBeCloseTo(600, 9);
    expect(fit.offsetY).toBeCloseTo(0, 9);
  });

  it("letterboxes a tall canvas around a wide image", () => {
    const fit = containFit(500, 1000, 1000, 500);
    expect(fit.scale).toBe(0.5);
    expect(fit.offsetX).toBe(0);
    expect(fit.offsetY).toBe(375);
    expect(fit.width).toBe(500);
    expect(fit.height).toBe(250);
  });

  it("fills the canvas exactly when the aspect ratios match", () => {
    const fit = containFit(1600, 1200, 800, 600);
    expect(fit).toEqual({ scale: 2, offsetX: 0, offsetY: 0, width: 1600, height: 1200 });
  });

  it("rejects non-positive sizes", () => {
    expect(() => containFit(0, 10, 10, 10)).toThrow(RangeError);
    expect(() => containFit(10, 10, 10, -1)).toThrow(RangeError);
    expect(() => containFit(10, NaN, 10, 10)).toThrow(RangeError);
  });
});

describe("canvas <-> image mapping", () => {
  const fit = containFit(1920, 1080, 1225, 1633);

  it("maps the image extent [-0.5, W-0.5] onto the displayed rectangle", () => {
    const [x0, y0] = imageToCanvas(fit, -0.5, -0.5);
    const [x1, y1] = imageToCanvas(fit, 1225 - 0.5, 1633 - 0.5);
    expect(x0).toBeCloseTo(fit.offsetX, 9);
    expect(y0).toBeCloseTo(fit.offsetY, 9);
    expect(x1).toBeCloseTo(fit.offsetX + fit.width, 9);
    expect(y1).toBeCloseTo(fit.offsetY + fit.height, 9);
  });

  it("round-trips in both directions", () => {
    for (let i = 0; i < 100; i += 1) {
      const x = (i * 37.31) % 1920;
      const y = (i * 91.7) % 1080;
      const [u, v] = canvasToImage(fit, x, y);
      const [x2, y2] = imageToCanvas(fit, u, v);
      expect(Math.abs(x2 - x)).toBeLessThan(1e-9);
      expect(Math.abs(y2 - y)).toBeLessThan(1e-9);
    }
  });

  it("puts the image centre at the canvas centre", () => {
    const [x, y] = imageToCanvas(fit, (1225 - 1) / 2, (1633 - 1) / 2);
    expect(x).toBeCloseTo(960, 9);
    expect(y).toBeCloseTo(540, 9);
  });

  it("knows the letterbox bars", () => {
    expect(insideImage(fit, 960, 540)).toBe(true);
    expect(insideImage(fit, 10, 540)).toBe(false);
    expect(insideImage(fit, fit.offsetX, 0)).toBe(true);
    expect(insideImage(fit, fit.offsetX + fit.width, 0)).toBe(false);
  });
});

describe("clipScale", () => {
  it("reproduces the pixel mapping at the identity pose", () => {
    // Project a few camera-space points with T08's pinhole, place them on the canvas with the contain-fit
    // mapping, and compare with what the shader computes: clip = k * (X, -Y) / Z, then the viewport transform.
    const W = 801;
    const H = 601;
    const f = 700;
    const [cw, ch] = [1920, 1080];
    const fit = containFit(cw, ch, W, H);
    const [kx, ky] = clipScale(fit, f, cw, ch);
    for (const [u, v, z] of [
      [0, 0, 3],
      [W - 1, H - 1, 7],
      [123.4, 400.2, 1.5],
      [(W - 1) / 2, (H - 1) / 2, 4],
    ] as const) {
      const [X, Y, Z] = unproject(u, v, z, f, W, H);
      const clipX = (kx * X) / Z;
      const clipY = (-ky * Y) / Z;
      const canvasX = ((clipX + 1) / 2) * cw;
      const canvasY = ((1 - clipY) / 2) * ch;
      const [ex, ey] = imageToCanvas(fit, u, v);
      expect(Math.abs(canvasX - ex)).toBeLessThan(1e-9);
      expect(Math.abs(canvasY - ey)).toBeLessThan(1e-9);
      // and the camera module's own projection of that point lands on the principal-point-relative pixel
      const [pu, pv] = project(X, Y, Z, f, W, H);
      expect(Math.abs(pu - u)).toBeLessThan(1e-9);
      expect(Math.abs(pv - v)).toBeLessThan(1e-9);
    }
  });
});

describe("mesh helpers", () => {
  it("derives square rows from the aspect", () => {
    expect(meshRows(1024, 2048, 1536)).toBe(768);
    expect(meshRows(1024, 1536, 2048)).toBe(1365);
    expect(meshRows(1, 10, 1)).toBe(1);
    expect(meshRows(10, 10000, 1)).toBe(1);
  });

  it("maps grid coordinates to depth texels like the shader's integer arithmetic", () => {
    expect(gridTexel(0, 1024, 2048)).toBe(0);
    expect(gridTexel(1, 1024, 2048)).toBe(2);
    expect(gridTexel(1024, 1024, 2048)).toBe(2047); // clamped: the far edge uses the last texel
    expect(gridTexel(3, 4, 10)).toBe(7);
    expect(gridTexel(0, 3, 7)).toBe(0);
    expect(gridTexel(1, 3, 7)).toBe(2);
  });

  it("flags quads whose corner depths span more than tau", () => {
    // 8x8 depth: left half 2 m, right half 4 m; a 4x4 mesh: the quad column at the step is cut
    const W = 8;
    const d = new Float32Array(W * W);
    for (let y = 0; y < W; y += 1) for (let x = 0; x < W; x += 1) d[y * W + x] = x < 4 ? 2 : 4;
    const cut = quadCutMask(d, W, W, 4, 4, 1.08);
    for (let y = 0; y < 4; y += 1) {
      expect(Array.from(cut.slice(y * 4, y * 4 + 4))).toEqual([0, 1, 0, 0]);
    }
    expect(quadCutMask(d, W, W, 4, 4, 5).some((c) => c === 1)).toBe(false);
  });
});

describe("fetchAssets", () => {
  afterEach(() => vi.unstubAllGlobals());

  const meta = {
    schema: 1,
    width: 8,
    height: 4,
    preview_width: 4,
    preview_height: 2,
    f_px_preview: 5,
    pivot_z: 3,
    scale: 0.5,
    depth: { max: 9 },
  };

  function stub(over: Record<string, () => Response> = {}): void {
    const depth = new Uint16Array(8).fill(floatToHalf(2));
    const files: Record<string, () => Response> = {
      "meta.json": () => new Response(JSON.stringify(meta)),
      "depth.f16": () => new Response(depth.buffer),
      "sky.u8": () => new Response(new Uint8Array(8)),
      "color.jpg": () => new Response(new Blob(["x"])),
      ...over,
    };
    vi.stubGlobal("fetch", (url: string) => {
      const name = url.split("/").pop() as string;
      return Promise.resolve(files[name]?.() ?? new Response("", { status: 404 }));
    });
  }
  const decode = () => Promise.resolve({ width: 4, height: 2 } as ImageBitmap);

  it("loads and validates the four files", async () => {
    stub();
    const a = await fetchAssets("/x", decode);
    expect(a.depth).toBeInstanceOf(Uint16Array);
    expect(a.depth).toHaveLength(8);
    expect(a.sky).toHaveLength(8);
    expect(a.meta.f_px_preview).toBe(5);
  });

  it("rejects wrong sizes, unknown schemas and HTTP errors", async () => {
    stub({ "depth.f16": () => new Response(new Uint8Array(6)) });
    await expect(fetchAssets("/x", decode)).rejects.toThrow(/depth.f16/);
    stub({ "sky.u8": () => new Response(new Uint8Array(3)) });
    await expect(fetchAssets("/x", decode)).rejects.toThrow(/sky.u8/);
    stub({ "meta.json": () => new Response(JSON.stringify({ ...meta, schema: 2 })) });
    await expect(fetchAssets("/x", decode)).rejects.toThrow(/schema/);
    stub({ "color.jpg": () => new Response("", { status: 500 }) });
    await expect(fetchAssets("/x", decode)).rejects.toThrow(/HTTP 500/);
  });
});
