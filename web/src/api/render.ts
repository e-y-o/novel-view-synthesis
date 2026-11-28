/** The `/render` endpoint, request building, stale detection and download file names. Pure and unit-tested. */
import { orbitToJson, type CameraSpec, type OrbitJson, type OrbitParams } from "../camera";
import { postJson } from "./client";

/** UI names for the hole options; `alpha` is what the API calls a transparent hole. */
export type HoleMode = "transparent" | "magenta" | "black";
const API_HOLE: Record<HoleMode, "alpha" | "magenta" | "black"> = {
  transparent: "alpha",
  magenta: "magenta",
  black: "black",
};

export interface RenderOptions {
  hole: HoleMode;
  fillSpecks: boolean;
}

export interface RenderBody {
  R: number[][];
  t: number[];
  orbit: OrbitJson;
  size: "source";
  hole: "alpha" | "magenta" | "black";
  fill_specks: boolean;
  format: "png";
}

export interface RenderResult {
  url: string;
  name: string;
  width: number;
  height: number;
  coverage: { disocclusion: number; stretch: number; out_of_frame: number; total: number };
  ms: Record<string, number>;
}

const r6 = (x: number): number => Math.round(x * 1e6) / 1e6 + 0; // "+ 0" turns -0 into 0

/** The request body: pose rounded to 1e-6 (far below the server's 1e-4 file naming), source size, PNG. */
export function buildRenderBody(
  camera: CameraSpec,
  orbit: OrbitParams,
  o: RenderOptions,
): RenderBody {
  return {
    R: camera.R.map((row) => row.map(r6)),
    t: camera.t.map(r6),
    orbit: orbitToJson(orbit),
    size: "source",
    hole: API_HOLE[o.hole],
    fill_specks: o.fillSpecks,
    format: "png",
  };
}

export function requestRender(
  id: string,
  body: RenderBody,
  signal?: AbortSignal,
): Promise<RenderResult> {
  return postJson<RenderResult>(`/api/sessions/${id}/render`, body, { signal });
}

/**
 * Identifies the pose a render was made for. Two orbits with the same key give the same image: rounded the same
 * way the server names result files (1e-4), plus the options that change the pixels.
 */
export function poseKey(orbit: OrbitParams, o: RenderOptions): string {
  const q = (x: number): string => (Math.round(x * 1e4) / 1e4 + 0).toFixed(4);
  return [
    q(orbit.yawDeg),
    q(orbit.pitchDeg),
    q(orbit.dolly),
    q(orbit.pan[0]),
    q(orbit.pan[1]),
    q(orbit.pivotZ),
    o.hole,
    o.fillSpecks ? "s" : "n",
  ].join("|");
}

/** A result is outdated when the view changed since it was rendered. */
export const isStale = (resultKey: string | null, currentKey: string | null): boolean =>
  resultKey !== null && currentKey !== null && resultKey !== currentKey;

const RESERVED = new Set([
  "con",
  "prn",
  "aux",
  "nul",
  ...[1, 2, 3, 4, 5, 6, 7, 8, 9].flatMap((i) => [`com${i}`, `lpt${i}`]),
]);

/** ASCII-only, Windows-safe stem: accents folded, illegal characters and reserved device names replaced. */
export function safeStem(filename: string, maxLength = 60): string {
  const withoutExt = filename.replace(/\.[^./\\]{1,8}$/, "");
  const ascii = withoutExt
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^A-Za-z0-9._ ()-]+/g, "_")
    .replace(/_{2,}/g, "_")
    .replace(/^[ ._]+|[ ._]+$/g, "")
    .slice(0, maxLength)
    .replace(/[ ._]+$/g, "");
  if (!ascii) return "image";
  return RESERVED.has(ascii.toLowerCase()) ? `_${ascii}` : ascii;
}

const signed = (x: number): string =>
  `${x < 0 || Object.is(x, -0) ? "-" : "+"}${Math.abs(x).toFixed(1)}`;

/**
 * `<original-stem>-view-yaw{+5.0}-pitch{+0.0}.png`. Yaw and pitch use the CLI convention (positive yaw moves
 * the camera right, positive pitch up), so the same view can be rendered with `nvs render --yaw 5 --pitch 0`.
 */
export function downloadName(filename: string, orbit: OrbitParams): string {
  // -0 after rounding is written as +0.0
  const yaw = Math.abs(orbit.yawDeg) < 0.05 ? 0 : orbit.yawDeg;
  const pitch = Math.abs(orbit.pitchDeg) < 0.05 ? 0 : orbit.pitchDeg;
  return `${safeStem(filename)}-view-yaw${signed(yaw)}-pitch${signed(pitch)}.png`;
}
