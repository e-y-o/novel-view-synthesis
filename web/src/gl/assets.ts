/** Fetch and decode the preview assets of a session (or of a generated fixture scene). */
import type { PreviewMetaLite, SessionAssets } from "./PreviewRenderer";

/** The `meta.json` fields this module reads (the file has more; unknown keys are ignored). */
export interface PreviewMeta extends PreviewMetaLite {
  schema: number;
  width: number;
  height: number;
  pivot_z: number;
  scale: number;
  filename?: string;
  f_px_source: number;
  intrinsics: { source: string; assumed: boolean; warnings: string[] };
  exif?: {
    make: string | null;
    model: string | null;
    f35: number | null;
    orientation: number | null;
  };
  warnings?: string[];
}

export const SUPPORTED_SCHEMA = 1;

export type BitmapDecoder = (blob: Blob) => Promise<ImageBitmap>;

async function getOk(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

/**
 * Load `color.jpg`, `depth.f16`, `sky.u8` and `meta.json` from `baseUrl` (no trailing slash), e.g.
 * `/api/sessions/<id>/assets` or `/fixtures/two_layers`. Checks the schema version and that every file has the
 * size `meta.json` promises. `decode` is injectable for tests.
 */
export async function fetchAssets(
  baseUrl: string,
  decode: BitmapDecoder = (blob) => createImageBitmap(blob),
): Promise<SessionAssets & { meta: PreviewMeta }> {
  const [metaRes, depthRes, skyRes, colorRes] = await Promise.all([
    getOk(`${baseUrl}/meta.json`),
    getOk(`${baseUrl}/depth.f16`),
    getOk(`${baseUrl}/sky.u8`),
    getOk(`${baseUrl}/color.jpg`),
  ]);
  const meta = (await metaRes.json()) as PreviewMeta;
  if (meta.schema !== SUPPORTED_SCHEMA) {
    throw new Error(
      `unsupported preview schema ${String(meta.schema)} (expected ${SUPPORTED_SCHEMA})`,
    );
  }
  const n = meta.preview_width * meta.preview_height;
  const depthBuf = await depthRes.arrayBuffer();
  if (depthBuf.byteLength !== n * 2) throw new Error("depth.f16 has the wrong size");
  const sky = new Uint8Array(await skyRes.arrayBuffer());
  if (sky.length !== n) throw new Error("sky.u8 has the wrong size");
  // Raw little-endian float16 bit patterns; every platform running a browser is little-endian.
  const depth = new Uint16Array(depthBuf);
  const color = await decode(await colorRes.blob());
  return { color, depth, sky, meta };
}
