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
  cutStats,
  fetchAssets,
  meshRows,
  quadCutMask,
  type CutStats,
  type DrawOptions,
  type SessionAssets,
} from "../src/gl";

const params = new URLSearchParams(location.search);
let scene = params.get("scene") ?? "two_layers";
const cols = Number(params.get("cols") ?? 1024);
const size = params.get("size") ?? "fit"; // fit | 1080p | 4k | native (canvas = preview size)
const num = (name: string, fallback: number): number => Number(params.get(name) ?? fallback);
const flag = (name: string, fallback: boolean): boolean =>
  params.has(name) ? params.get(name) !== "0" : fallback;
const urlOptions: Partial<DrawOptions> = {
  cutMode: params.get("mode") === "tau" ? "tau" : "stretch",
  cutTau: num("cut", 1.08),
  stretchFade: num("fade", 1.6),
  stretchCut: num("cutat", 3),
  stretchSmoothFactor: num("smooth", 2.5),
  feather: flag("feather", true),
  skyBoundaryCut: flag("skycut", false),
  showSky: flag("sky", true),
};
if (params.get("bg") === "magenta") urlOptions.background = [1, 0, 1];
if (params.get("bg") === "gray") urlOptions.background = [0.5, 0.5, 0.5];
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
let assets = await fetchAssets(`/fixtures/${scene}`);
await renderer.loadAssets(assets);
renderer.setOptions(urlOptions);
window.addEventListener("resize", () => renderer.resize());

function fitCanvasToImage(): void {
  if (size !== "native") return;
  renderer.setFixedSize([assets.meta.preview_width, assets.meta.preview_height]);
}
fitCanvasToImage();

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
function optionsLabel(): string {
  const o = currentOptions();
  return o.cutMode === "tau"
    ? `tau ${o.cutTau}`
    : `stretch ${o.stretchFade}..${o.stretchCut}${o.feather ? " feather" : ""}${o.skyBoundaryCut ? " skycut" : ""}`;
}
setInterval(() => {
  const s = renderer.stats();
  errorLog = errorLog.concat(renderer.glErrors());
  const rows = meshRows(cols, assets.meta.preview_width, assets.meta.preview_height);
  hud.textContent =
    `${scene}  ${assets.meta.preview_width}x${assets.meta.preview_height}  mesh ${cols}x${rows}\n` +
    `canvas ${canvas.width}x${canvas.height}  ${optionsLabel()}\n` +
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
  const cut = quadCutMask(decodeHalfArray(assets.depth), W, H, cols, rows, 1.08);
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

// ---- A/B tools ----------------------------------------------------------------------------------------
const merged: Partial<DrawOptions> = { ...urlOptions };
function currentOptions(): DrawOptions {
  return {
    cutMode: "stretch",
    cutTau: 1.08,
    stretchFade: 1.6,
    stretchCut: 3,
    stretchSmoothFactor: 2.5,
    feather: true,
    skyBoundaryCut: false,
    showSky: true,
    background: [0.14, 0.14, 0.14],
    ...merged,
  };
}
function setOptions(o: Partial<DrawOptions>): void {
  Object.assign(merged, o);
  renderer.setOptions(o);
}

/** Switch fixture scene without reloading the page (`name` is relative to /fixtures). */
async function loadScene(name: string): Promise<void> {
  scene = name;
  assets = await fetchAssets(`/fixtures/${name}`);
  await renderer.loadAssets(assets);
  Object.assign(orbit, defaultOrbit(assets.meta.pivot_z));
  fitCanvasToImage();
  apply();
}

/** Canvas pixels as RGBA with the top row first. */
function readCanvas(): Uint8Array {
  const w = canvas.width;
  const h = canvas.height;
  const raw = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, raw);
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y += 1) {
    out.set(raw.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
  }
  return out;
}

/** Render `yaw/pitch` with the options, over two background colours; a pixel is "filled" if both agree. */
function renderFilled(
  yaw: number,
  pitch: number,
  options: Partial<DrawOptions> = {},
): { rgb: Uint8Array; filled: Uint8Array } {
  const keep = { ...orbit };
  orbit.yawDeg = yaw;
  orbit.pitchDeg = pitch;
  const saved = currentOptions();
  renderer.setOptions({ ...options, background: [0, 0, 0] });
  renderer.setCamera(orbitPose(orbit));
  renderer.render();
  const a = readCanvas();
  renderer.setOptions({ background: [1, 1, 1] });
  renderer.render();
  const b = readCanvas();
  renderer.setOptions({ ...saved });
  Object.assign(orbit, keep);
  apply();
  const filled = new Uint8Array(a.length / 4);
  for (let i = 0; i < filled.length; i += 1) {
    const d = Math.max(
      Math.abs((a[i * 4] as number) - (b[i * 4] as number)),
      Math.abs((a[i * 4 + 1] as number) - (b[i * 4 + 1] as number)),
      Math.abs((a[i * 4 + 2] as number) - (b[i * 4 + 2] as number)),
    );
    filled[i] = d <= 2 ? 1 : 0;
  }
  return { rgb: a, filled };
}

async function loadReference(yaw: number, pitch: number): Promise<Uint8ClampedArray> {
  const url = `/fixtures/${scene}/ref_${yaw}_${pitch}.png`;
  const bitmap = await createImageBitmap(await (await fetch(url)).blob());
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext("2d") as OffscreenCanvasRenderingContext2D;
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
}

/**
 * Compare the preview with the Python splat of the same pose (`scripts/preview_reference.py`; needs
 * `size=native`). Returns the mean absolute error (0..255) over pixels both renderers fill, and the fractions.
 */
async function compareWithReference(
  yaw: number,
  pitch = 0,
  options: Partial<DrawOptions> = {},
  baseline?: Partial<DrawOptions>,
): Promise<Record<string, number>> {
  const W = assets.meta.preview_width;
  const H = assets.meta.preview_height;
  if (canvas.width !== W || canvas.height !== H) {
    throw new Error(
      `canvas ${canvas.width}x${canvas.height} != preview ${W}x${H}: use size=native`,
    );
  }
  const ref = await loadReference(yaw, pitch);
  const { rgb, filled } = renderFilled(yaw, pitch, options);
  // `baseline`: also report the error on the pixels the baseline variant fills (a like-for-like population)
  const baseFilled = baseline ? renderFilled(yaw, pitch, baseline).filled : null;
  let sum = 0;
  let baseSum = 0;
  let baseCount = 0;
  let both = 0;
  let browser = 0;
  let python = 0;
  let large = 0;
  const m = 2; // ignore a 2 px border (the splat's footprint at the frame edge)
  for (let y = m; y < H - m; y += 1) {
    for (let x = m; x < W - m; x += 1) {
      const i = y * W + x;
      const pyValid = (ref[i * 4 + 3] as number) > 127;
      const brValid = filled[i] === 1;
      if (pyValid) python += 1;
      if (brValid) browser += 1;
      if (!pyValid || !brValid) continue;
      both += 1;
      let e = 0;
      for (let c = 0; c < 3; c += 1)
        e += Math.abs((rgb[i * 4 + c] as number) - (ref[i * 4 + c] as number));
      sum += e / 3;
      if (baseFilled && baseFilled[i] === 1) {
        baseSum += e / 3;
        baseCount += 1;
      }
      if (e / 3 > 40) large += 1;
    }
  }
  const total = (W - 2 * m) * (H - 2 * m);
  return {
    mae255: sum / Math.max(both, 1),
    maeOnBaselineSet: baseCount ? baseSum / baseCount : NaN,
    bothFraction: both / total,
    browserFilled: browser / total,
    pythonFilled: python / total,
    largeErrorFraction: large / Math.max(both, 1),
  };
}

/** Removed / feathered quad counts from the CPU reference at a pose, for the loaded scene. */
function quadStats(yaw: number, pitch = 0, options: Partial<DrawOptions> = {}): CutStats {
  const W = assets.meta.preview_width;
  const H = assets.meta.preview_height;
  const o = { ...currentOptions(), ...options };
  return cutStats(
    decodeHalfArray(assets.depth),
    assets.sky,
    { width: W, height: H, cols, rows: meshRows(cols, W, H), f: assets.meta.f_px_preview },
    orbitPose({ ...orbit, yawDeg: yaw, pitchDeg: pitch }),
    o,
  );
}

Object.assign(window, {
  __renderer: renderer,
  __dev: {
    checkIdentity,
    runBench,
    leakCheck,
    contextLossCheck,
    compareWithReference,
    quadStats,
    loadScene,
    setOptions,
    renderFilled,
    orbit,
    get assets() {
      return assets;
    },
  },
});
apply();
