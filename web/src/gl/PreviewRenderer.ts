/**
 * Raw WebGL2 preview of a photo as a depth-displaced grid mesh (no three.js).
 *
 * Technique (proved by the M0 spike): a grid of non-indexed quads is drawn with no vertex attributes; the
 * vertex shader derives quad and corner from `gl_VertexID`, reads depths with `texelFetch` from an `R16F`
 * texture, unprojects with the source intrinsics and projects with the target pose. Quads whose corners span
 * a large depth ratio are cut (`discard`), which shows the background colour through the hole. Drawing is on
 * demand: `requestRender()` sets a dirty flag and schedules one `requestAnimationFrame`.
 *
 * The renderer owns every GL resource and frees it in `dispose()`. It does not close the `ImageBitmap` it is
 * given (it keeps it to restore the textures after a lost context). See `layout.ts` for the canvas/image
 * coordinate mapping.
 */
import type { CameraSpec } from "../camera";
import { identityPose } from "../camera";
import { clipScale, containFit, meshRows, type ContainFit } from "./layout";
import { canvasToImage as toImage, imageToCanvas as toCanvas } from "./layout";
import fragSrc from "./shaders/preview.frag.glsl?raw";
import vertSrc from "./shaders/preview.vert.glsl?raw";

/** The part of the server's preview `meta.json` the renderer needs. */
export interface PreviewMetaLite {
  preview_width: number;
  preview_height: number;
  /** Focal length in preview pixels. */
  f_px_preview: number;
  depth: { max: number };
}

/** Decoded preview assets (see the README contract). */
export interface SessionAssets {
  /** sRGB colour, `preview_width × preview_height`, oriented. Not closed by the renderer. */
  color: ImageBitmap;
  /** Raw float16 bit patterns of `depth.f16` (metres), row-major. */
  depth: Uint16Array;
  /** `sky.u8`: 255 sky, 0 otherwise, same size. */
  sky: Uint8Array;
  meta: PreviewMetaLite;
}

export interface RendererOptions {
  /** Mesh columns; rows follow from the aspect. 1024 is "2×" density. Default 1024. */
  meshCols?: number;
  /** Enable GPU timer queries and `getError` checks after each render. Default false. */
  debug?: boolean;
  /** Passed to `getContext`. Default false (MSAA is not needed and costs fill rate). */
  antialias?: boolean;
}

export interface DrawOptions {
  /** Cut threshold on max(z)/min(z) over a quad's corners. Default 1.08 (T10 makes this adaptive). */
  cutTau: number;
  /** Draw pixels the sky mask marks as sky. Default true. */
  showSky: boolean;
  /** Colour shown where the mesh is cut or absent, linear 0..1 RGB. Default neutral dark grey. */
  background: [number, number, number];
}

export interface RendererStats {
  /** CPU time of the last `render()` call, ms (command submission, not GPU time). */
  drawMs: number | null;
  /** Median GPU time of recent frames in ms; only with `debug` and `EXT_disjoint_timer_query_webgl2`. */
  gpuMs: number | null;
  /** Frames per second over recent frames, ignoring idle gaps longer than 500 ms. */
  fps: number;
  /** Number of draws since creation. */
  frames: number;
}

/** WebGL2, `texelFetch` or `R16F` textures are unavailable. `message` is meant for the user. */
export class RendererUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RendererUnsupportedError";
  }
}

/** A shader failed to compile or link; the message carries the driver's info log. */
export class ShaderError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShaderError";
  }
}

const DEFAULTS: DrawOptions = { cutTau: 1.08, showSky: true, background: [0.14, 0.14, 0.14] };
const NEAR = 0.02;
const FRAME_WINDOW = 120;
const IDLE_GAP_MS = 500;

interface Resources {
  program: WebGLProgram;
  vao: WebGLVertexArrayObject;
  depth: WebGLTexture;
  color: WebGLTexture;
  sky: WebGLTexture;
  uniforms: Record<string, WebGLUniformLocation | null>;
}

const UNIFORMS = [
  "uDepth",
  "uColor",
  "uSky",
  "uCols",
  "uRows",
  "uImgI",
  "uImg",
  "uF",
  "uRt",
  "uT",
  "uClipScale",
  "uTau",
  "uNear",
  "uFar",
  "uShowSky",
] as const;

export class PreviewRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly meshCols: number;
  private readonly debug: boolean;
  private options: DrawOptions = { ...DEFAULTS, background: [...DEFAULTS.background] };
  private camera: CameraSpec = identityPose();
  private assets: SessionAssets | null = null;
  private res: Resources | null = null;
  private fit: ContainFit | null = null;
  private lost = false;
  private disposed = false;
  private dirty = false;
  private raf = 0;
  private frames = 0;
  private lastDrawMs: number | null = null;
  private frameStamps: number[] = [];
  private timer: {
    ext: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number };
    pending: WebGLQuery | null;
    samples: number[];
  } | null = null;
  private readonly onLost = (e: Event): void => {
    e.preventDefault();
    this.lost = true;
    this.res = null; // the GPU objects are gone with the context
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    if (this.timer) this.timer.pending = null;
  };
  private readonly onRestored = (): void => {
    this.lost = false;
    try {
      this.res = this.createResources();
      if (this.assets) this.uploadAssets(this.assets);
      this.resize();
    } catch (e) {
      console.error("PreviewRenderer: could not restore after context loss", e);
    }
  };

  constructor(canvas: HTMLCanvasElement, options: RendererOptions = {}) {
    this.canvas = canvas;
    this.meshCols = options.meshCols ?? 1024;
    this.debug = options.debug ?? false;
    const gl = canvas.getContext("webgl2", {
      antialias: options.antialias ?? false,
      powerPreference: "high-performance",
    });
    if (!gl) {
      throw new RendererUnsupportedError(
        "This browser cannot run the 3D preview: WebGL 2 is not available. Try a current Chrome, Edge, Firefox or Safari with hardware acceleration enabled.",
      );
    }
    this.gl = gl;
    canvas.addEventListener("webglcontextlost", this.onLost);
    canvas.addEventListener("webglcontextrestored", this.onRestored);
    this.res = this.createResources();
    if (this.debug) {
      const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
      if (ext) this.timer = { ext, pending: null, samples: [] };
    }
    this.resize();
  }

  // ------------------------------------------------------------------------------------ resources
  private createResources(): Resources {
    const gl = this.gl;
    const program = gl.createProgram();
    if (!program) throw new RendererUnsupportedError("Could not create a WebGL program.");
    const vs = this.compile(gl.VERTEX_SHADER, vertSrc);
    const fs = this.compile(gl.FRAGMENT_SHADER, fragSrc);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? "";
      gl.deleteProgram(program);
      throw new ShaderError(`shader link failed: ${log}`);
    }
    const uniforms: Record<string, WebGLUniformLocation | null> = {};
    for (const name of UNIFORMS) uniforms[name] = gl.getUniformLocation(program, name);
    const vao = gl.createVertexArray();
    const depth = gl.createTexture();
    const color = gl.createTexture();
    const sky = gl.createTexture();
    if (!vao || !depth || !color || !sky) {
      throw new RendererUnsupportedError("Could not allocate WebGL objects (out of GPU memory?).");
    }
    return { program, vao, depth, color, sky, uniforms };
  }

  private compile(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new RendererUnsupportedError("Could not create a WebGL shader.");
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) ?? "";
      gl.deleteShader(shader);
      throw new ShaderError(`shader compile failed: ${log}`);
    }
    return shader;
  }

  private uploadAssets(a: SessionAssets): void {
    const gl = this.gl;
    const res = this.res;
    if (!res) return;
    const { preview_width: W, preview_height: H } = a.meta;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); // R16F / R8 rows are not 4-byte aligned in general
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); // the Y flip happens in the shader
    while (gl.getError() !== gl.NO_ERROR) {
      /* drain stale errors so the upload check below is ours */
    }

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, res.depth);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, W, H, 0, gl.RED, gl.HALF_FLOAT, a.depth);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    if (gl.getError() !== gl.NO_ERROR) {
      throw new RendererUnsupportedError(
        "This GPU or browser cannot use 16-bit float textures, which the 3D preview needs.",
      );
    }

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, res.color);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, a.color);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); // no mipmaps
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, res.sky);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, W, H, 0, gl.RED, gl.UNSIGNED_BYTE, a.sky);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // ------------------------------------------------------------------------------------------ API
  /**
   * Upload new assets, replacing the textures of any earlier call (no leak). Validates sizes first.
   * @throws RendererUnsupportedError if the GPU cannot use float16 textures; `RangeError` on bad assets.
   */
  loadAssets(assets: SessionAssets): Promise<void> {
    this.assertLive();
    const { preview_width: W, preview_height: H, f_px_preview: f } = assets.meta;
    if (!(W > 0 && H > 0 && f > 0))
      throw new RangeError("meta must have positive size and focal length");
    if (assets.depth.length !== W * H)
      throw new RangeError("depth.f16 does not match the preview size");
    if (assets.sky.length !== W * H) throw new RangeError("sky.u8 does not match the preview size");
    if (assets.color.width !== W || assets.color.height !== H) {
      throw new RangeError("color image does not match the preview size");
    }
    this.assets = assets;
    this.uploadAssets(assets);
    this.fit = null;
    this.resize();
    return Promise.resolve();
  }

  /** Target camera pose in the source frame (`P_target = Rᵀ (P − t)`, T08). Triggers a redraw. */
  setCamera(camera: CameraSpec): void {
    this.camera = camera;
    this.requestRender();
  }

  /** Change drawing options. Unspecified fields keep their value. Triggers a redraw. */
  setOptions(options: Partial<DrawOptions>): void {
    this.options = {
      ...this.options,
      ...options,
      background: options.background ? [...options.background] : this.options.background,
    };
    this.requestRender();
  }

  /** Match the canvas backing store to its CSS size × `devicePixelRatio` and refit the image. */
  resize(): void {
    const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
    const max = this.gl.getParameter(this.gl.MAX_VIEWPORT_DIMS) as Int32Array;
    const w = Math.min(Math.max(1, Math.round(this.canvas.clientWidth * dpr)), max[0] ?? 4096);
    const h = Math.min(Math.max(1, Math.round(this.canvas.clientHeight * dpr)), max[1] ?? 4096);
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    this.fit = this.assets
      ? containFit(w, h, this.assets.meta.preview_width, this.assets.meta.preview_height)
      : null;
    this.requestRender();
  }

  /** Schedule one draw on the next animation frame (coalesces repeated calls; idle = no work). */
  requestRender(): void {
    this.dirty = true;
    if (this.raf || this.disposed || this.lost) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      if (this.dirty) this.render();
    });
  }

  /** Draw now. Does nothing before `loadAssets`, while the context is lost, or after `dispose`. */
  render(): void {
    this.dirty = false;
    const res = this.res;
    const a = this.assets;
    const fit = this.fit;
    if (this.disposed || this.lost || !res || !a || !fit) return;
    const gl = this.gl;
    const t0 = performance.now();
    const { preview_width: W, preview_height: H, f_px_preview: f } = a.meta;
    const rows = meshRows(this.meshCols, W, H);
    const [kx, ky] = clipScale(fit, f, this.canvas.width, this.canvas.height);
    const u = res.uniforms;
    const bg = this.options.background;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(res.program);
    gl.bindVertexArray(res.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, res.depth);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, res.color);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, res.sky);
    gl.uniform1i(u.uDepth ?? null, 0);
    gl.uniform1i(u.uColor ?? null, 1);
    gl.uniform1i(u.uSky ?? null, 2);
    gl.uniform1i(u.uCols ?? null, this.meshCols);
    gl.uniform1i(u.uRows ?? null, rows);
    gl.uniform2i(u.uImgI ?? null, W, H);
    gl.uniform2f(u.uImg ?? null, W, H);
    gl.uniform1f(u.uF ?? null, f);
    // GLSL mat3 data is column-major: uploading R row-major without transpose yields Rᵀ in the shader.
    const R = this.camera.R;
    gl.uniformMatrix3fv(u.uRt ?? null, false, new Float32Array([...R[0], ...R[1], ...R[2]]));
    gl.uniform3f(u.uT ?? null, this.camera.t[0], this.camera.t[1], this.camera.t[2]);
    gl.uniform2f(u.uClipScale ?? null, kx, ky);
    gl.uniform1f(u.uTau ?? null, this.options.cutTau);
    gl.uniform1f(u.uNear ?? null, NEAR);
    gl.uniform1f(u.uFar ?? null, Math.max(a.meta.depth.max * 4, 50));
    gl.uniform1i(u.uShowSky ?? null, this.options.showSky ? 1 : 0);

    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    let query: WebGLQuery | null = null;
    const timer = this.timer;
    if (timer && !timer.pending) {
      query = gl.createQuery();
      if (query) gl.beginQuery(timer.ext.TIME_ELAPSED_EXT, query);
    }
    gl.drawArrays(gl.TRIANGLES, 0, this.meshCols * rows * 6);
    if (timer && query) {
      gl.endQuery(timer.ext.TIME_ELAPSED_EXT);
      timer.pending = query;
    }
    if (timer) this.pollTimer();

    const now = performance.now();
    this.lastDrawMs = now - t0;
    this.frames += 1;
    this.frameStamps.push(now);
    if (this.frameStamps.length > FRAME_WINDOW) this.frameStamps.shift();
  }

  private pollTimer(): void {
    const timer = this.timer;
    const q = timer?.pending;
    if (!timer || !q) return;
    const gl = this.gl;
    if (gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) {
      const disjoint = gl.getParameter(timer.ext.GPU_DISJOINT_EXT) as boolean;
      if (!disjoint) {
        timer.samples.push(Number(gl.getQueryParameter(q, gl.QUERY_RESULT)) / 1e6);
        if (timer.samples.length > FRAME_WINDOW) timer.samples.shift();
      }
      gl.deleteQuery(q);
      timer.pending = null;
    }
  }

  /** Drain and return pending WebGL error codes (empty when clean). Cheap enough for tests and the dev page. */
  glErrors(): number[] {
    const errors: number[] = [];
    for (let i = 0; i < 16; i += 1) {
      const e = this.gl.getError();
      if (e === this.gl.NO_ERROR) break;
      errors.push(e);
    }
    return errors;
  }

  stats(): RendererStats {
    if (this.timer) this.pollTimer();
    let fps = 0;
    const s = this.frameStamps;
    if (s.length >= 2) {
      let total = 0;
      let n = 0;
      for (let i = 1; i < s.length; i += 1) {
        const dt = (s[i] as number) - (s[i - 1] as number);
        if (dt <= IDLE_GAP_MS) {
          total += dt;
          n += 1;
        }
      }
      fps = n > 0 && total > 0 ? (1000 * n) / total : 0;
    }
    const samples = [...(this.timer?.samples ?? [])].sort((x, y) => x - y);
    const gpuMs = samples.length ? (samples[Math.floor(samples.length / 2)] as number) : null;
    return { drawMs: this.lastDrawMs, gpuMs, fps, frames: this.frames };
  }

  /** Canvas pixel (device pixels) to preview-image pixel; `null` before assets are loaded. */
  canvasToImage(x: number, y: number): [number, number] | null {
    return this.fit ? toImage(this.fit, x, y) : null;
  }

  /** Preview-image pixel to canvas pixel (device pixels); `null` before assets are loaded. */
  imageToCanvas(u: number, v: number): [number, number] | null {
    return this.fit ? toCanvas(this.fit, u, v) : null;
  }

  /** The current contain-fit, or `null` before assets are loaded. */
  get layout(): ContainFit | null {
    return this.fit;
  }

  /** True while the WebGL context is lost (drawing is suspended until it is restored). */
  get contextLost(): boolean {
    return this.lost;
  }

  /** Release every GL resource and listener. The renderer is unusable afterwards. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.canvas.removeEventListener("webglcontextlost", this.onLost);
    this.canvas.removeEventListener("webglcontextrestored", this.onRestored);
    const gl = this.gl;
    const res = this.res;
    if (res && !this.lost) {
      gl.deleteTexture(res.depth);
      gl.deleteTexture(res.color);
      gl.deleteTexture(res.sky);
      gl.deleteVertexArray(res.vao);
      gl.deleteProgram(res.program);
      if (this.timer?.pending) gl.deleteQuery(this.timer.pending);
    }
    this.res = null;
    this.assets = null;
  }

  private assertLive(): void {
    if (this.disposed) throw new Error("PreviewRenderer has been disposed");
  }
}
