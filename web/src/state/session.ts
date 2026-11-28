/**
 * Application state (signals) and the flows that change it: upload, analysis progress, opening, deleting.
 * UI components read the signals; only the functions here write them.
 */
import { batch, signal } from "@preact/signals";
import { ApiError } from "../api/client";
import { connectEvents, type EventsConnection, type ServerEvent } from "../api/events";
import {
  deleteSession,
  getSession,
  listSessions,
  uploadSession,
  type SessionDetail,
  type SessionSummary,
} from "../api/sessions";
import type { PreviewMeta } from "../gl/assets";
import { describeError, sessionHash, validateFile, type ErrorInfo } from "../ui/helpers";

export type Phase = "upload" | "decode" | "intrinsics" | "depth" | "package";
export const PHASES: { id: Phase; label: string }[] = [
  { id: "upload", label: "Uploading" },
  { id: "decode", label: "Reading the photo" },
  { id: "intrinsics", label: "Finding the focal length" },
  { id: "depth", label: "Estimating depth" },
  { id: "package", label: "Preparing the preview" },
];

export type ViewState = "idle" | "uploading" | "analyzing" | "ready";

export interface View {
  state: ViewState;
  id: string | null;
  filename: string;
  phase: Phase;
  /** Progress inside the current phase, 0..100. */
  pct: number;
  meta: PreviewMeta | null;
  /** Object URL of the user's own photo, shown blurred while waiting. */
  blurUrl: string | null;
}

export interface AppError extends ErrorInfo {
  code: string;
  status: number;
  /** Text for "Copy details". */
  details: string;
}

export type WorkerState = { state: "starting" | "ready" | "error" | "unknown"; detail: string };

const IDLE: View = {
  state: "idle",
  id: null,
  filename: "",
  phase: "upload",
  pct: 0,
  meta: null,
  blurUrl: null,
};

export const view = signal<View>({ ...IDLE });
export const worker = signal<WorkerState>({ state: "unknown", detail: "" });
export const recent = signal<SessionSummary[]>([]);
export const appError = signal<AppError | null>(null);
/** Polite screen-reader announcements (progress phases, errors). */
export const announcement = signal("");
export const meshQuality = signal<1 | 2>(2);

let sessionEvents: EventsConnection | null = null;
let globalEvents: EventsConnection | null = null;
let lastFile: File | null = null;
let attempt = 0;
let abort: AbortController | null = null;

// ----------------------------------------------------------------------------------------- errors
export function showError(code: string, status = 0, message = "", id: string | null = null): void {
  const info = describeError(code, status, message);
  const details = [
    `code: ${code}`,
    `http status: ${status || "n/a"}`,
    `message: ${message || "n/a"}`,
    `session: ${id ?? view.value.id ?? "n/a"}`,
    `attempt: ${attempt}`,
    `time: ${new Date().toISOString()}`,
  ].join("\n");
  batch(() => {
    appError.value = { ...info, code, status, details };
    announcement.value = `${info.title}. ${info.message}`;
  });
}

export function dismissError(): void {
  appError.value = null;
}

function fail(e: unknown): void {
  if (e instanceof ApiError) {
    if (e.code === "aborted") return;
    showError(e.code, e.status, e.message);
  } else {
    showError("unknown", 0, e instanceof Error ? e.message : String(e));
  }
  const v = view.value;
  if (v.state === "uploading" || v.state === "analyzing") setView({ ...IDLE });
}

// ------------------------------------------------------------------------------------------- view
function setView(next: View): void {
  const old = view.value.blurUrl;
  if (old && old !== next.blurUrl) URL.revokeObjectURL(old);
  view.value = next;
}

function patchView(p: Partial<View>): void {
  const next = { ...view.value, ...p };
  if (view.value.blurUrl && next.blurUrl === null) URL.revokeObjectURL(view.value.blurUrl);
  view.value = next;
}

/** Free the blurred own-photo once the viewer has faded in. */
export function clearBlur(): void {
  if (view.value.blurUrl) patchView({ blurUrl: null });
}

export function goHome(): void {
  closeSessionEvents();
  abort?.abort();
  setView({ ...IDLE });
  dismissError();
  if (location.hash && location.hash !== "#/") history.replaceState(null, "", "#/");
}

// --------------------------------------------------------------------------------------- events
export function startGlobalEvents(): () => void {
  globalEvents?.close();
  globalEvents = connectEvents("/api/events", {
    onEvent: (e) => {
      if (e.type === "status") worker.value = { state: e.worker, detail: e.detail };
    },
    onClose: () => {
      worker.value = { state: "unknown", detail: "" };
    },
  });
  return () => {
    globalEvents?.close();
    globalEvents = null;
  };
}

function closeSessionEvents(): void {
  sessionEvents?.close();
  sessionEvents = null;
}

/** Apply one server event for the session that is currently open (exported for tests). */
export function handleSessionEvent(e: ServerEvent, id: string): void {
  if (e.type === "status") {
    worker.value = { state: e.worker, detail: e.detail };
    return;
  }
  if (view.value.id !== id) return;
  if (e.type === "analyze.progress") {
    const phase = PHASES.some((p) => p.id === e.phase) ? (e.phase as Phase) : view.value.phase;
    const changed = phase !== view.value.phase;
    patchView({ state: "analyzing", phase, pct: e.pct });
    if (changed) announcement.value = PHASES.find((p) => p.id === phase)?.label ?? "";
  } else if (e.type === "ready") {
    void finishAnalysis(id);
  } else if (e.type === "error") {
    showError(e.code, 0, e.message, id);
    patchView({ state: "idle" });
  }
}

function watchSession(id: string): void {
  closeSessionEvents();
  sessionEvents = connectEvents(`/api/sessions/${id}/events`, {
    onEvent: (e) => handleSessionEvent(e, id),
  });
}

async function finishAnalysis(id: string): Promise<void> {
  try {
    const detail = await getSession(id);
    if (view.value.id !== id) return;
    patchView({
      state: "ready",
      meta: detail.meta,
      filename: detail.filename || view.value.filename,
    });
    announcement.value = "Preview ready. Drag to orbit.";
    void loadRecent();
  } catch (e) {
    fail(e);
  }
}

// ------------------------------------------------------------------------------------------ flows
/** Upload a photo and follow its analysis. */
export async function startUpload(file: File): Promise<void> {
  const check = validateFile(file);
  if (!check.ok) {
    showError("unsupported_type", 0, check.message);
    return;
  }
  lastFile = file;
  attempt += 1;
  dismissError();
  abort?.abort();
  abort = new AbortController();
  setView({
    ...IDLE,
    state: "uploading",
    filename: file.name,
    blurUrl: URL.createObjectURL(file),
  });
  announcement.value = `Uploading ${file.name}`;
  try {
    const result = await uploadSession(
      file,
      (pct) => view.value.state === "uploading" && patchView({ pct }),
      abort.signal,
    );
    patchView({ id: result.id, filename: result.filename || file.name });
    history.replaceState(null, "", sessionHash(result.id));
    if (result.state === "ready") {
      await finishAnalysis(result.id);
      return;
    }
    patchView({ state: "analyzing", phase: "decode", pct: 0 });
    watchSession(result.id);
  } catch (e) {
    fail(e);
  }
}

/** Retry the last upload (after a worker or GPU error). */
export function retryUpload(): void {
  if (lastFile) void startUpload(lastFile);
}

/** Open an existing session (hash route or Recent list). */
export async function openSession(id: string): Promise<void> {
  if (view.value.id === id && view.value.state !== "idle") return;
  closeSessionEvents();
  dismissError();
  try {
    const detail: SessionDetail = await getSession(id);
    setView({
      ...IDLE,
      id,
      filename: detail.filename,
      meta: detail.meta,
      state: detail.state === "ready" ? "ready" : "analyzing",
    });
    if (location.hash !== sessionHash(id)) history.replaceState(null, "", sessionHash(id));
    if (detail.state === "ready") return;
    if (detail.state === "error" && detail.error) {
      showError(detail.error.code, 0, detail.error.message, id);
      setView({ ...IDLE });
      return;
    }
    patchView({ phase: "decode", pct: 0 });
    watchSession(id); // the hub replays the latest progress
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      showError("not_found", 404, e.message, id);
      setView({ ...IDLE });
      history.replaceState(null, "", "#/");
    } else {
      fail(e);
    }
  }
}

export async function loadRecent(): Promise<void> {
  try {
    recent.value = await listSessions();
  } catch {
    /* the list is optional: a failure must not hide the app */
  }
}

export async function removeSession(id: string): Promise<void> {
  try {
    await deleteSession(id);
    if (view.value.id === id) goHome();
    await loadRecent();
  } catch (e) {
    fail(e);
  }
}
