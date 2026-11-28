/**
 * Dev page for `PreviewRenderer`: loads a fixture scene, orbit controls, HUD (fps, GPU ms), and scripted
 * checks exposed on `window.__renderer` / `window.__dev` for manual and automated use (T15 Playwright).
 */
import {
  decodeHalfArray,
  defaultOrbit,
  identityPose,
  orbitPose,
  type OrbitParams,
} from "../src/camera";
import {
  PreviewRenderer,
  RendererUnsupportedError,
  fetchAssets,
  meshRows,
  quadCutMask,
  type SessionAssets,
} from "../src/gl";

const params = new URLSearchParams(location.search);
const scene = params.get("scene") ?? "two_layers";
const cols = Number(params.get("cols") ?? 1024);
const tau = Number(params.get("cut") ?? 1.08);
const size = params.get("size") ?? "fit";
const SIZES: Record<string, [number, number]> = { "1080p": [1920, 1080], "4k": [3840, 2160] };

const canvas = document.getElementById("gl") as HTMLCanvasElement;
const hud = document.getElementById("hud") as HTMLDivElement;

// Fixed backing-store sizes for the benchmarks: CSS size = target / devicePixelRatio.
const fixed = SIZES[size];
if (fixed) {
  canvas.style.width = `${fixed[0] / devicePixelRatio}px`;
  canvas.style.height = `${fixed[1] / devicePixelRatio}px`;
} else {
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
}

let renderer: PreviewRenderer;
try {
  renderer = new PreviewRenderer(canvas, { meshCols: cols, debug: params.get("debug") !== "0" });
} catch (e) {
  hud.textContent = e instanceof RendererUnsupportedError ? e.message : String(e);
  throw e;
}
const assets = await fetchAssets(`/fixtures/${scene}`);
await renderer.loadAssets(assets);
renderer.setOptions({ cutTau: tau });
window.addEventListener("resize", () => renderer.resize());

// ---- orbit state --------------------------------------------------------------------------------------
const orbit: OrbitParams = defaultOrbit(assets.meta.pivot_z);
function apply(): void {
  renderer.setCamera(orbitPose(orbit));
}
let bench = params.get("bench") === "1";
let dragging = false;
let lx = 0;
let ly = 0;
canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  lx = e.clientX;
  ly = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointerup", (e) => {
  dragging = false;
  canvas.releasePointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging || bench) return;
  orbit.yawDeg = Math.max(-45, Math.min(45, orbit.yawDeg - (e.clientX - lx) * 0.2));
  orbit.pitchDeg = Math.max(-30, Math.min(30, orbit.pitchDeg + (e.clientY - ly) * 0.2));
  lx = e.clientX;
  ly = e.clientY;
  apply();
});
canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const p = orbit.pivotZ;
    orbit.dolly = Math.max(-p * 0.5, Math.min(p * 0.8, orbit.dolly - e.deltaY * 0.001 * p));
    apply();
  },
  { passive: false },
);
canvas.addEventListener("dblclick", () => {
  Object.assign(orbit, defaultOrbit(orbit.pivotZ));
  apply();
});
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    bench = !bench;
    e.preventDefault();
  }
});

// ---- benchmark loop: auto-orbit +-10 deg yaw / +-4 deg pitch, a draw every animation frame ------------
function benchFrame(now: number): void {
  if (bench) {
    const s = now / 1000;
    orbit.yawDeg = 10 * Math.sin(s * 1.3);
    orbit.pitchDeg = 4 * Math.sin(s * 0.9);
    apply();
  }
  requestAnimationFrame(benchFrame);
}
requestAnimationFrame(benchFrame);

// ---- HUD ----------------------------------------------------------------------------------------------
const gl = canvas.getContext("webgl2") as WebGL2RenderingContext;
const dbg = gl.getExtension("WEBGL_debug_renderer_info");
const gpuName = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : "unknown GPU";
let errorLog: number[] = [];
setInterval(() => {
  const s = renderer.stats();
  errorLog = errorLog.concat(renderer.glErrors());
  const rows = meshRows(cols, assets.meta.preview_width, assets.meta.preview_height);
  hud.textContent =
    `${scene}  ${assets.meta.preview_width}x${assets.meta.preview_height}  mesh ${cols}x${rows}\n` +
    `canvas ${canvas.width}x${canvas.height}  cut ${tau}\n` +
    `fps ${bench ? s.fps.toFixed(1) : "(idle: draws on change)"}  gpu ${s.gpuMs === null ? "n/a" : s.gpuMs.toFixed(2) + " ms"}  cpu ${s.drawMs?.toFixed(2) ?? "n/a"} ms\n` +
    `yaw ${orbit.yawDeg.toFixed(1)} pitch ${orbit.pitchDeg.toFixed(1)} dolly ${orbit.dolly.toFixed(2)}\n` +
    `GL errors: ${errorLog.length ? errorLog.join(",") : "none"}\n${gpuName}`;
}, 250);

// ---- scripted checks ----------------------------------------------------------------------------------
interface IdentityReport {
  meanAbsErr255: number;
  comparedPixels: number;
  skippedNearCut: number;
  maxAbsErr: number;
  canvas: [number, number];
  glErrors: number[];
}

/** Render the identity pose and compare with the colour image resampled to the canvas (contain fit). */
function checkIdentity(): IdentityReport {
  const prev = { ...orbit };
  Object.assign(orbit, defaultOrbit(orbit.pivotZ));
  renderer.setCamera(identityPose());
  renderer.render();
  const w = canvas.width;
  const h = canvas.height;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const glErrors = renderer.glErrors();

  const ref = new OffscreenCanvas(w, h);
  const ctx = ref.getContext("2d") as OffscreenCanvasRenderingContext2D;
  const fit = renderer.layout;
  if (!fit) throw new Error("no layout");
  ctx.fillStyle = "#242424";
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(assets.color, fit.offsetX, fit.offsetY, fit.width, fit.height);
  const want = ctx.getImageData(0, 0, w, h).data;

  const W = assets.meta.preview_width;
  const H = assets.meta.preview_height;
  const rows = meshRows(cols, W, H);
  const cut = quadCutMask(decodeHalfArray(assets.depth), W, H, cols, rows, tau);
  const nearCut = (x: number, y: number): boolean => {
    const qx = Math.floor((((x - fit.offsetX) / fit.scale) * cols) / W);
    const qy = Math.floor((((y - fit.offsetY) / fit.scale) * rows) / H);
    for (let dy = -1; dy <= 1; dy += 1)
      for (let dx = -1; dx <= 1; dx += 1) {
        const cx = qx + dx;
        const cy = qy + dy;
        if (cx >= 0 && cx < cols && cy >= 0 && cy < rows && cut[cy * cols + cx]) return true;
      }
    return false;
  };

  let sum = 0;
  let n = 0;
  let skipped = 0;
  let max = 0;
  const m = 3; // stay away from the image border
  for (let y = Math.ceil(fit.offsetY) + m; y < Math.floor(fit.offsetY + fit.height) - m; y += 1) {
    for (let x = Math.ceil(fit.offsetX) + m; x < Math.floor(fit.offsetX + fit.width) - m; x += 1) {
      if (nearCut(x, y)) {
        skipped += 1;
        continue;
      }
      const gi = ((h - 1 - y) * w + x) * 4; // readPixels rows start at the bottom
      const ri = (y * w + x) * 4;
      for (let c = 0; c < 3; c += 1) {
        const d = Math.abs((px[gi + c] as number) - (want[ri + c] as number));
        sum += d;
        if (d > max) max = d;
      }
      n += 1;
    }
  }
  Object.assign(orbit, prev);
  apply();
  return {
    meanAbsErr255: sum / (3 * Math.max(n, 1)),
    comparedPixels: n,
    skippedNearCut: skipped,
    maxAbsErr: max,
    canvas: [w, h],
    glErrors,
  };
}

/** Auto-orbit for `seconds`, returning the renderer's fps and GPU-time statistics. */
async function runBench(seconds = 6): Promise<Record<string, unknown>> {
  bench = true;
  await new Promise((r) => setTimeout(r, 1500)); // warm-up
  const gpu: number[] = [];
  const t0 = performance.now();
  while (performance.now() - t0 < seconds * 1000) {
    await new Promise((r) => requestAnimationFrame(() => r(null)));
    const g = renderer.stats().gpuMs;
    if (g !== null) gpu.push(g);
  }
  const s = renderer.stats();
  bench = false;
  gpu.sort((a, b) => a - b);
  const pick = (p: number): number | null =>
    gpu[Math.min(gpu.length - 1, Math.floor(p * gpu.length))] ?? null;
  return {
    canvas: [canvas.width, canvas.height],
    mesh: `${cols}x${meshRows(cols, assets.meta.preview_width, assets.meta.preview_height)}`,
    fps: s.fps,
    gpuMsMedian: s.gpuMs,
    gpuMsP50: pick(0.5),
    gpuMsP95: pick(0.95),
    gpuSamples: gpu.length,
    errors: renderer.glErrors(),
    gpu: gpuName,
  };
}

/** Count texture/program/VAO creations and deletions across reloads and dispose (leak check). */
async function leakCheck(): Promise<Record<string, number>> {
  const counts = { created: 0, deleted: 0 };
  const proto = WebGL2RenderingContext.prototype;
  const original = {
    createTexture: proto.createTexture,
    deleteTexture: proto.deleteTexture,
    createProgram: proto.createProgram,
    deleteProgram: proto.deleteProgram,
    createVertexArray: proto.createVertexArray,
    deleteVertexArray: proto.deleteVertexArray,
  };
  proto.createTexture = function (this: WebGL2RenderingContext) {
    counts.created += 1;
    return original.createTexture.call(this);
  };
  proto.deleteTexture = function (this: WebGL2RenderingContext, t: WebGLTexture | null) {
    counts.deleted += 1;
    original.deleteTexture.call(this, t);
  };
  proto.createProgram = function (this: WebGL2RenderingContext) {
    counts.created += 1;
    return original.createProgram.call(this);
  };
  proto.deleteProgram = function (this: WebGL2RenderingContext, p: WebGLProgram | null) {
    counts.deleted += 1;
    original.deleteProgram.call(this, p);
  };
  proto.createVertexArray = function (this: WebGL2RenderingContext) {
    counts.created += 1;
    return original.createVertexArray.call(this);
  };
  proto.deleteVertexArray = function (
    this: WebGL2RenderingContext,
    v: WebGLVertexArrayObject | null,
  ) {
    counts.deleted += 1;
    original.deleteVertexArray.call(this, v);
  };
  try {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 256;
    const r = new PreviewRenderer(c, { meshCols: 64 });
    const a: SessionAssets = assets;
    await r.loadAssets(a);
    const afterFirst = counts.created;
    await r.loadAssets(a); // replaces the textures: no new objects
    await r.loadAssets(a);
    const createdByReloads = counts.created - afterFirst;
    r.dispose();
    return { created: counts.created, deleted: counts.deleted, createdByReloads };
  } finally {
    Object.assign(proto, original);
  }
}

/** Lose and restore the context; returns whether drawing works again afterwards. */
async function contextLossCheck(): Promise<Record<string, unknown>> {
  const c = document.createElement("canvas");
  c.style.width = "320px";
  c.style.height = "240px";
  document.body.appendChild(c);
  try {
    const r = new PreviewRenderer(c, { meshCols: 128 });
    await r.loadAssets(assets);
    r.render();
    const ext = (c.getContext("webgl2") as WebGL2RenderingContext).getExtension(
      "WEBGL_lose_context",
    );
    if (!ext) return { supported: false };
    ext.loseContext();
    await new Promise((res) => setTimeout(res, 100));
    const lost = r.contextLost;
    r.render(); // must be a harmless no-op
    ext.restoreContext();
    await new Promise((res) => setTimeout(res, 300));
    r.render();
    const errors = r.glErrors();
    r.dispose();
    return { supported: true, lost, restored: !r.contextLost, errors };
  } finally {
    c.remove();
  }
}

Object.assign(window, {
  __renderer: renderer,
  __dev: { checkIdentity, runBench, leakCheck, contextLossCheck, orbit, assets },
});
apply();
