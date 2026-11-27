import vertSrc from "./mesh.vert.glsl?raw";
import fragSrc from "./mesh.frag.glsl?raw";

interface Meta { width: number; height: number; f_px: number; z_min: number; z_max: number; pivot_z: number }

const params = new URLSearchParams(location.search);
const scene = params.get("scene") ?? "architecture_01";
const cols = Number(params.get("cols") ?? 512);
const tau = Number(params.get("cut") ?? 1.08);
const sizeParam = params.get("size") ?? "fit"; // "1080p" | "4k" | "fit"
const SIZES: Record<string, [number, number]> = { "1080p": [1920, 1080], "4k": [3840, 2160] };

const canvas = document.getElementById("gl") as HTMLCanvasElement;
const hud = document.getElementById("hud") as HTMLDivElement;
const gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance" })!;
if (!gl) throw new Error("WebGL2 not available");

function compile(type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
  return s;
}
const prog = gl.createProgram()!;
gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertSrc));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragSrc));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? "link error");
gl.useProgram(prog);
const U = (n: string) => gl.getUniformLocation(prog, n);

const dbg = gl.getExtension("WEBGL_debug_renderer_info");
const rendererName = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : "unknown";
const timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");

// ---- assets -------------------------------------------------------------------------------------------
async function load(): Promise<{ meta: Meta; depth: Uint16Array; color: ImageBitmap }> {
  const base = `/${scene}/`;
  const meta: Meta = await (await fetch(base + "meta.json")).json();
  const buf = await (await fetch(base + "depth.f16")).arrayBuffer();
  const depth = new Uint16Array(buf); // raw little-endian float16 bit patterns (x86/ARM are little-endian)
  if (depth.length !== meta.width * meta.height) throw new Error("depth.f16 size does not match meta.json");
  const color = await createImageBitmap(await (await fetch(base + "color.jpg")).blob());
  return { meta, depth, color };
}

function makeTexture(unit: number): WebGLTexture {
  const t = gl.createTexture()!;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, t);
  return t;
}

const { meta, depth, color } = await load();
const W = meta.width, H = meta.height;
const rows = Math.round((cols * H) / W);

gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); // R16F rows are W*2 bytes, not a multiple of 4 in general
makeTexture(0);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, W, H, 0, gl.RED, gl.HALF_FLOAT, depth);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); // half-float: no linear filtering needed (texelFetch)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
makeTexture(1);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, color);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.uniform1i(U("uDepth"), 0);
gl.uniform1i(U("uColor"), 1);

const vao = gl.createVertexArray()!; // no attributes: positions come from gl_VertexID
gl.bindVertexArray(vao);

// ---- canvas size --------------------------------------------------------------------------------------
const [cw, ch] = SIZES[sizeParam] ?? [Math.round((H / 1) * (W / H)), H]; // "fit": backing store = image size
canvas.width = sizeParam in SIZES ? cw : W;
canvas.height = sizeParam in SIZES ? ch : H;

gl.uniform1i(U("uCols"), cols);
gl.uniform1i(U("uRows"), rows);
gl.uniform2f(U("uImg"), W, H);
gl.uniform1f(U("uF"), meta.f_px);
gl.uniform1f(U("uTau"), tau);
gl.uniform1f(U("uAspect"), canvas.height / canvas.width);
gl.uniform1f(U("uNear"), 0.05);
gl.uniform1f(U("uFar"), meta.z_max * 2);
gl.enable(gl.DEPTH_TEST);
gl.clearColor(0.125, 0.14, 0.16, 1);
gl.viewport(0, 0, canvas.width, canvas.height);

// ---- camera: orbit about (0,0,pivot) ------------------------------------------------------------------
// Target pose in the source frame: P_cam = R^T (P - t), R = Ry(yaw) Rx(pitch), camera keeps looking at the pivot.
const cam = { yaw: 0, pitch: 0, dolly: 0 }; // degrees, degrees, metres moved forward along the optical axis
const pivot = meta.pivot_z;

function pose(): { R: number[]; t: [number, number, number] } {
  const a = (cam.yaw * Math.PI) / 180, b = (cam.pitch * Math.PI) / 180;
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const Ry = [ca, 0, sa, 0, 1, 0, -sa, 0, ca];
  const Rx = [1, 0, 0, 0, cb, -sb, 0, sb, cb];
  const R = new Array<number>(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) R[i * 3 + j] += Ry[i * 3 + k] * Rx[k * 3 + j];
  // t = p - R (p - (0,0,dolly)) : orbit about p, then move along the optical axis by `dolly`
  const q = [0, 0, pivot - cam.dolly];
  const t: [number, number, number] = [
    -(R[0] * q[0] + R[1] * q[1] + R[2] * q[2]),
    -(R[3] * q[0] + R[4] * q[1] + R[5] * q[2]),
    pivot - (R[6] * q[0] + R[7] * q[1] + R[8] * q[2]),
  ];
  return { R, t };
}

// ---- render loop (only while dirty, or continuously in benchmark mode) --------------------------------
let dirty = true;
let bench = params.get("bench") === "1";
let frames = 0, lastHud = performance.now(), fps = 0, gpuMs = NaN;
const frameTimes: number[] = [];
let pendingQuery: WebGLQuery | null = null;
const gpuSamples: number[] = [];

function draw(): void {
  const { R, t } = pose();
  // uRt = R^T. GLSL mat3 data is column-major: passing R's row-major array unchanged yields its transpose.
  gl.uniformMatrix3fv(U("uRt"), false, R);
  gl.uniform3f(U("uT"), t[0], t[1], t[2]);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  let q: WebGLQuery | null = null;
  if (timerExt && !pendingQuery) { q = gl.createQuery()!; gl.beginQuery(timerExt.TIME_ELAPSED_EXT, q); }
  gl.drawArrays(gl.TRIANGLES, 0, cols * rows * 6);
  if (q && timerExt) { gl.endQuery(timerExt.TIME_ELAPSED_EXT); pendingQuery = q; }
}

function pollQuery(): void {
  if (!pendingQuery || !timerExt) return;
  const ok = gl.getQueryParameter(pendingQuery, gl.QUERY_RESULT_AVAILABLE);
  const disjoint = gl.getParameter(timerExt.GPU_DISJOINT_EXT);
  if (ok && !disjoint) {
    gpuMs = Number(gl.getQueryParameter(pendingQuery, gl.QUERY_RESULT)) / 1e6;
    gpuSamples.push(gpuMs);
    gl.deleteQuery(pendingQuery);
    pendingQuery = null;
  } else if (ok) { gl.deleteQuery(pendingQuery); pendingQuery = null; }
}

let last = performance.now();
function frame(now: number): void {
  if (bench) { // auto-orbit +-10 deg yaw, +-4 deg pitch
    const s = now / 1000;
    cam.yaw = 10 * Math.sin(s * 1.3);
    cam.pitch = 4 * Math.sin(s * 0.9);
    dirty = true;
  }
  if (dirty) {
    draw();
    dirty = false;
    frames++;
    frameTimes.push(now - last);
    if (frameTimes.length > 600) frameTimes.shift();
  }
  last = now;
  pollQuery();
  if (now - lastHud > 500) {
    fps = (frames * 1000) / (now - lastHud);
    frames = 0; lastHud = now;
    hud.textContent =
      `${scene}  ${W}x${H}\n` +
      `quads ${(cols * rows).toLocaleString()} (${cols}x${rows}), ${(cols * rows * 6).toLocaleString()} verts\n` +
      `canvas ${canvas.width}x${canvas.height}  cut ${tau}\n` +
      `fps ${bench ? fps.toFixed(1) : "(idle: renders on change)"}  gpu ${Number.isNaN(gpuMs) ? "n/a" : gpuMs.toFixed(2) + " ms"}\n` +
      `yaw ${cam.yaw.toFixed(1)} pitch ${cam.pitch.toFixed(1)} dolly ${cam.dolly.toFixed(2)} m\n` +
      `${rendererName}`;
  }
  requestAnimationFrame(frame);
}
// The rAF callback is cheap when nothing is dirty; draw() itself only runs on change.
requestAnimationFrame(frame);

// ---- input --------------------------------------------------------------------------------------------
let dragging = false, lx = 0, ly = 0;
canvas.addEventListener("pointerdown", (e) => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointerup", (e) => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", (e) => {
  if (!dragging || bench) return;
  const scale = 0.2; // degrees per CSS pixel
  cam.yaw -= (e.clientX - lx) * scale;   // drag right -> content moves right -> negative yaw
  cam.pitch += (e.clientY - ly) * scale; // drag down -> content moves down
  cam.yaw = Math.max(-45, Math.min(45, cam.yaw));
  cam.pitch = Math.max(-30, Math.min(30, cam.pitch));
  lx = e.clientX; ly = e.clientY; dirty = true;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  cam.dolly = Math.max(-pivot * 0.5, Math.min(pivot * 0.8, cam.dolly - e.deltaY * 0.001 * pivot));
  dirty = true;
}, { passive: false });
canvas.addEventListener("dblclick", () => { cam.yaw = 0; cam.pitch = 0; cam.dolly = 0; dirty = true; });
window.addEventListener("keydown", (e) => { if (e.code === "Space") { bench = !bench; if (!bench) dirty = true; e.preventDefault(); } });

// Hooks for scripted measurement from the browser tool.
(window as unknown as Record<string, unknown>).__nvs = {
  setCam(yaw: number, pitch = 0, dolly = 0) { cam.yaw = yaw; cam.pitch = pitch; cam.dolly = dolly; dirty = true; },
  setBench(on: boolean) { bench = on; dirty = true; },
  stats() {
    const ft = [...frameTimes].sort((a, b) => a - b);
    const gs = [...gpuSamples].sort((a, b) => a - b);
    const pick = (a: number[], p: number) => (a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : null);
    const mean = ft.length ? ft.reduce((a, b) => a + b, 0) / ft.length : NaN;
    return { frames: ft.length, fps_mean: 1000 / mean, frame_ms_p50: pick(ft, 0.5), frame_ms_p95: pick(ft, 0.95), frame_ms_max: ft[ft.length - 1] ?? null,
      gpu_ms_p50: pick(gs, 0.5), gpu_ms_p95: pick(gs, 0.95), gpu_samples: gs.length, renderer: rendererName, canvas: [canvas.width, canvas.height], quads: cols * rows };
  },
  async capture(name: string) { // dev server only: see vite.config.ts
    draw();
    const blob: Blob = await new Promise((r) => canvas.toBlob((b) => r(b!), "image/png"));
    return (await fetch(`/__save?name=${name}`, { method: "POST", body: blob })).text();
  },
  reset() { frameTimes.length = 0; gpuSamples.length = 0; },
};
