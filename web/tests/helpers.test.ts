import { describe, expect, it } from "vitest";
import {
  describeError,
  equivalentMm,
  exifChip,
  firstImageFile,
  parseHash,
  sessionHash,
  validateFile,
} from "../src/ui/helpers";

const file = (name: string, type: string, size = 10) => ({ name, type, size }) as File;

describe("validateFile", () => {
  it("accepts the supported types by MIME or by extension", () => {
    for (const [name, type] of [
      ["a.jpg", "image/jpeg"],
      ["a.JPEG", ""],
      ["a.png", "image/png"],
      ["a.webp", "image/webp"],
      ["IMG_1.HEIC", ""], // browsers often report no type for HEIC
      ["a.heif", "image/heif"],
      ["noext", "image/jpeg"],
    ] as const) {
      expect(validateFile(file(name, type)).ok, name).toBe(true);
    }
  });

  it("rejects other types with a message that names the file and the formats", () => {
    for (const [name, type] of [
      ["notes.txt", "text/plain"],
      ["clip.mp4", "video/mp4"],
      ["scan.pdf", "application/pdf"],
      ["pic.gif", "image/gif"],
      ["pic.tiff", "image/tiff"],
    ] as const) {
      const r = validateFile(file(name, type));
      expect(r.ok, name).toBe(false);
      if (!r.ok) expect(r.message).toContain(name);
      if (!r.ok) expect(r.message).toContain("JPEG, PNG, WebP or HEIC");
    }
    const empty = validateFile(file("a.jpg", "image/jpeg", 0));
    expect(empty.ok).toBe(false);
  });

  it("picks the first image among dropped or pasted files", () => {
    const a = file("a.txt", "text/plain");
    const b = file("b.png", "image/png");
    expect(firstImageFile([a, b])).toBe(b);
    expect(firstImageFile([a])).toBeNull();
    expect(firstImageFile(null)).toBeNull();
    expect(firstImageFile([file("x", "image/gif")])?.name).toBe("x"); // any image/* is tried; the server decides
  });
});

describe("hash routing", () => {
  const id = "ab12".repeat(16);
  it("round-trips a session id", () => {
    expect(sessionHash(id)).toBe(`#/s/${id}`);
    expect(parseHash(sessionHash(id))).toEqual({ kind: "session", id });
    expect(parseHash(`${sessionHash(id)}/`)).toEqual({ kind: "session", id });
  });
  it("treats everything else as home", () => {
    for (const h of [
      "",
      "#",
      "#/",
      "#/s/",
      "#/s/xyz",
      `#/s/${id.slice(1)}`,
      `#/s/${id.toUpperCase()}`,
      "#/other",
    ]) {
      expect(parseHash(h), h).toEqual({ kind: "home" });
    }
  });
});

describe("describeError", () => {
  it("has a specific message and next step for every known code", () => {
    const table: [string, RegExp, string][] = [
      ["gpu_oom", /GPU memory/, "retry"],
      ["worker_died", /worker/i, "retry"],
      ["worker_unavailable", /not ready/i, "retry"],
      ["licence_not_accepted", /--accept-license/, "none"],
      ["unsupported_image", /could not be read/, "choose"],
      ["depth_failed", /Depth estimation failed/, "retry"],
      ["timeout", /too long/, "retry"],
      ["webgl_unsupported", /WebGL/, "none"],
      ["network", /nvs serve/, "retry"],
      ["not_found", /no longer exists/, "choose"],
    ];
    for (const [code, text, action] of table) {
      const e = describeError(code);
      expect(e.title.length, code).toBeGreaterThan(3);
      expect(`${e.title} ${e.message}`, code).toMatch(text);
      expect(e.action, code).toBe(action);
    }
  });

  it("names the file for a type rejected in the browser", () => {
    const e = describeError(
      "unsupported_type",
      0,
      '"notes.txt" is not a supported image. Use JPEG, PNG, WebP or HEIC.',
    );
    expect(e.title).toBe("Not a supported image");
    expect(e.message).toContain("notes.txt");
    expect(e.action).toBe("choose");
    expect(describeError("unsupported_type").message).toContain("JPEG");
  });

  it("falls back on the HTTP status, then on the server message", () => {
    expect(describeError("http_415", 415).title).toMatch(/could not be read/);
    expect(describeError("http_413", 413).message).toMatch(/200 MB/);
    expect(describeError("http_404", 404).action).toBe("choose");
    const other = describeError("weird", 500, "the server said no");
    expect(other.message).toBe("the server said no");
    expect(other.action).toBe("retry");
    expect(describeError("weird", 0, "").message.length).toBeGreaterThan(10);
  });
});

describe("EXIF chip", () => {
  // a 4000x3000 image whose 35 mm-equivalent focal length is 26 mm: f_px = 26 * 5000 / 43.267
  const fPx = (26 * 5000) / 43.267;
  const base = { f_px_source: fPx, width: 4000, height: 3000 };

  it("converts pixels to 35 mm-equivalent", () => {
    expect(equivalentMm(fPx, 4000, 3000)).toBeCloseTo(26, 9);
    expect(equivalentMm(1000, 3000, 4000)).toBeCloseTo(equivalentMm(1000, 4000, 3000), 12);
  });

  it("is calm for EXIF sources", () => {
    const a = exifChip({ ...base, intrinsics: { source: "35mm", assumed: false, warnings: [] } });
    expect(a).toEqual({ text: "f = 26 mm eq. · from EXIF", tone: "ok", details: [] });
    const b = exifChip({
      ...base,
      intrinsics: { source: "focal_plane", assumed: false, warnings: ["w"] },
    });
    expect(b.text).toBe("f = 26 mm eq. · from EXIF (sensor size)");
    expect(b.tone).toBe("ok");
    expect(b.details).toEqual(["w"]);
  });

  it("is amber when the focal length is estimated or assumed, with the reason", () => {
    const est = exifChip({
      ...base,
      intrinsics: { source: "estimator", assumed: true, warnings: ["x"] },
    });
    expect(est.tone).toBe("warn");
    expect(est.text).toBe("f ≈ 26 mm eq. · estimated");
    expect(est.details.at(-1)).toBe("x");
    expect(est.details[0]).toMatch(/estimated/);
    const def = exifChip({
      ...base,
      intrinsics: { source: "default", assumed: true, warnings: [] },
    });
    expect(def.tone).toBe("warn");
    expect(def.text).toBe("focal length assumed");
    expect(def.details[0]).toMatch(/assumed/);
  });

  it("shows a user override as set by the user", () => {
    const o = exifChip({
      ...base,
      intrinsics: { source: "override", assumed: false, warnings: [] },
    });
    expect(o.text).toBe("f = 26 mm eq. · set by you");
  });
});
