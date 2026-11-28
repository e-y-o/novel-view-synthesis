/** Pure helpers for the shell: file validation, hash routing, error text, EXIF chip. Unit-tested. */
import type { PreviewMeta } from "../gl/assets";

// ------------------------------------------------------------------------------------------- files
export const ACCEPT = "image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif";
export const FORMATS_TEXT = "JPEG, PNG, WebP or HEIC";
const EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"];
const MIME = new Set(["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]);

export type FileCheck = { ok: true } | { ok: false; message: string };

/** Accept by MIME type or, when the browser reports none (HEIC often), by extension. */
export function validateFile(file: Pick<File, "name" | "type" | "size">): FileCheck {
  const name = file.name.toLowerCase();
  const byExt = EXTENSIONS.some((e) => name.endsWith(e));
  const byMime = MIME.has(file.type.toLowerCase());
  if (!byExt && !byMime) {
    return {
      ok: false,
      message: `"${file.name}" is not a supported image. Use ${FORMATS_TEXT}.`,
    };
  }
  if (file.size === 0) return { ok: false, message: `"${file.name}" is empty.` };
  return { ok: true };
}

/** The first image among pasted or dropped items, if any. */
export function firstImageFile(files: Iterable<File> | null | undefined): File | null {
  if (!files) return null;
  for (const f of files) if (validateFile(f).ok || f.type.startsWith("image/")) return f;
  return null;
}

// ------------------------------------------------------------------------------------------ routing
export type Route = { kind: "home" } | { kind: "session"; id: string };
const SESSION_ID = /^[0-9a-f]{64}$/;

/** `#/s/<id>` identifies a session; anything else is the home page. */
export function parseHash(hash: string): Route {
  const m = /^#\/s\/([0-9a-f]+)\/?$/.exec(hash);
  if (m && SESSION_ID.test(m[1] as string)) return { kind: "session", id: m[1] as string };
  return { kind: "home" };
}

export const sessionHash = (id: string): string => `#/s/${id}`;

// ---------------------------------------------------------------------------------------- errors
export interface ErrorInfo {
  title: string;
  message: string;
  /** The next step the banner offers. */
  action: "retry" | "choose" | "none";
}

const BY_CODE: Record<string, ErrorInfo> = {
  gpu_oom: {
    title: "Not enough GPU memory",
    message: "Close other GPU applications (browsers and games use video memory) and try again.",
    action: "retry",
  },
  worker_died: {
    title: "The depth worker stopped",
    message:
      "The process that estimates depth crashed. It restarts by itself; try again in a few seconds.",
    action: "retry",
  },
  worker_unavailable: {
    title: "The depth model is not ready",
    message: "The model is still loading or could not start. Wait a moment and retry.",
    action: "retry",
  },
  licence_not_accepted: {
    title: "Model licence not accepted",
    message:
      "Depth Pro 2 is research-only. Stop the server and start it again with `nvs serve --accept-license` to accept the licence.",
    action: "none",
  },
  unsupported_image: {
    title: "This file could not be read",
    message: `It is not a valid image, or it is damaged. Use ${FORMATS_TEXT}.`,
    action: "choose",
  },
  depth_failed: {
    title: "Depth estimation failed",
    message: "The model could not process this photo. Try another photo or retry.",
    action: "retry",
  },
  timeout: {
    title: "Analysis took too long",
    message: "The photo was not processed in time. Try again, or use a smaller photo.",
    action: "retry",
  },
  webgl_unsupported: {
    title: "3D preview not available",
    message:
      "This browser cannot run the 3D preview (WebGL 2 or 16-bit float textures are missing). Try a current Chrome, Edge, Firefox or Safari with hardware acceleration on.",
    action: "none",
  },
  network: {
    title: "Cannot reach the server",
    message: "Check that `nvs serve` is still running, then retry.",
    action: "retry",
  },
  not_found: {
    title: "Photo not found",
    message: "This session no longer exists. It may have been deleted. Choose the photo again.",
    action: "choose",
  },
};

/** Map an API/worker error to the banner text. Falls back on the HTTP status, then on the raw message. */
export function describeError(code: string, status = 0, message = ""): ErrorInfo {
  const known = BY_CODE[code];
  if (known) return known;
  if (status === 415) return BY_CODE["unsupported_image"] as ErrorInfo;
  if (status === 413) {
    return {
      title: "File too large",
      message: "The photo exceeds the size limit (200 MB, 100 megapixels). Use a smaller export.",
      action: "choose",
    };
  }
  if (status === 404) return BY_CODE["not_found"] as ErrorInfo;
  return {
    title: "Something went wrong",
    message:
      message || "An unexpected error occurred. Retry; if it keeps happening, copy the details.",
    action: "retry",
  };
}

// ----------------------------------------------------------------------------------------- EXIF chip
export interface ExifChip {
  text: string;
  /** `warn` (amber) when the focal length is a guess. */
  tone: "ok" | "warn";
  /** Tooltip lines: warnings from `meta.intrinsics.warnings`. */
  details: string[];
}

const FULL_FRAME_DIAG_MM = 43.267;

/** 35 mm-equivalent focal length in mm of `fPx` pixels on a `width × height` image. */
export function equivalentMm(fPx: number, width: number, height: number): number {
  return (fPx * FULL_FRAME_DIAG_MM) / Math.hypot(width, height);
}

type ChipMeta = Pick<PreviewMeta, "f_px_source" | "width" | "height" | "intrinsics">;

export function exifChip(meta: ChipMeta): ExifChip {
  const mm = Math.round(equivalentMm(meta.f_px_source, meta.width, meta.height));
  const { source, assumed, warnings } = meta.intrinsics;
  const details = [...warnings];
  switch (source) {
    case "35mm":
      return { text: `f = ${mm} mm eq. · from EXIF`, tone: "ok", details };
    case "focal_plane":
      return { text: `f = ${mm} mm eq. · from EXIF (sensor size)`, tone: "ok", details };
    case "override":
      return { text: `f = ${mm} mm eq. · set by you`, tone: "ok", details };
    case "estimator":
      return {
        text: `f ≈ ${mm} mm eq. · estimated`,
        tone: "warn",
        details: ["No focal length in the file: it was estimated from the image.", ...details],
      };
    default:
      return {
        text: assumed ? "focal length assumed" : `f = ${mm} mm eq.`,
        tone: assumed ? "warn" : "ok",
        details: assumed
          ? [
              "No focal length in the file and no estimate: a typical phone lens was assumed.",
              ...details,
            ]
          : details,
      };
  }
}
