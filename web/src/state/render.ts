/**
 * Full-resolution render: state and the single-flight job. One render at a time; a second click while one runs
 * does nothing; cancelling aborts the request, which makes the server cancel the job.
 */
import { signal } from "@preact/signals";
import { ApiError } from "../api/client";
import {
  buildRenderBody,
  poseKey,
  requestRender,
  type RenderBody,
  type RenderOptions,
  type RenderResult,
} from "../api/render";
import { describeError, type ErrorInfo } from "../ui/helpers";
import type { CameraSpec, OrbitParams } from "../camera";

export interface RenderOutcome {
  result: RenderResult;
  /** `poseKey` of the view that was rendered, to tell when it becomes outdated. */
  key: string;
  orbit: OrbitParams;
  options: RenderOptions;
  filename: string;
}

export type RenderStatus = "idle" | "running" | "done" | "error";

export interface RenderState {
  status: RenderStatus;
  /** `performance.now()` when the running render started. */
  startedAt: number;
  /** The last finished render; kept (and marked outdated) while a new one runs. */
  outcome: RenderOutcome | null;
  error: (ErrorInfo & { code: string }) | null;
}

const EMPTY: RenderState = { status: "idle", startedAt: 0, outcome: null, error: null };

export const renderState = signal<RenderState>({ ...EMPTY });
export const renderOptions = signal<RenderOptions>({ hole: "transparent", fillSpecks: true });
/** Show the finished render over the preview (the Preview / Result toggle). */
export const showResult = signal(false);
/** Space held: compare with the result without changing the toggle. */
export const holdCompare = signal(false);
/** Elapsed seconds while a render runs (for the spinner label). */
export const elapsed = signal(0);

let abort: AbortController | null = null;
let ticker: ReturnType<typeof setInterval> | null = null;

export interface RenderDeps {
  send: (id: string, body: RenderBody, signal: AbortSignal) => Promise<RenderResult>;
  now: () => number;
}
const defaultDeps: RenderDeps = { send: requestRender, now: () => performance.now() };

function stopTicker(): void {
  if (ticker) clearInterval(ticker);
  ticker = null;
}

/**
 * Render the current view at the photo's resolution. Returns false (and does nothing) if a render is already
 * running, so double clicks and key repeats cannot start two.
 */
export async function startRender(
  id: string,
  camera: CameraSpec,
  orbit: OrbitParams,
  filename: string,
  deps: RenderDeps = defaultDeps,
): Promise<boolean> {
  if (renderState.value.status === "running") return false;
  const options = renderOptions.value;
  const body = buildRenderBody(camera, orbit, options);
  const key = poseKey(orbit, options);
  abort = new AbortController();
  const mine = abort;
  const startedAt = deps.now();
  renderState.value = { ...renderState.value, status: "running", startedAt, error: null };
  elapsed.value = 0;
  stopTicker();
  ticker = setInterval(() => (elapsed.value = (deps.now() - startedAt) / 1000), 100);
  try {
    const result = await deps.send(id, body, mine.signal);
    if (mine.signal.aborted) return true;
    renderState.value = {
      status: "done",
      startedAt,
      outcome: {
        result,
        key,
        orbit: { ...orbit, pan: [orbit.pan[0], orbit.pan[1]] },
        options: { ...options },
        filename,
      },
      error: null,
    };
    showResult.value = true;
  } catch (e) {
    if (mine.signal.aborted || (e instanceof DOMException && e.name === "AbortError")) {
      renderState.value = {
        ...renderState.value,
        status: renderState.value.outcome ? "done" : "idle",
      };
    } else {
      const code = e instanceof ApiError ? e.code : "unknown";
      const info = describeError(
        code,
        e instanceof ApiError ? e.status : 0,
        e instanceof Error ? e.message : "",
      );
      renderState.value = { ...renderState.value, status: "error", error: { ...info, code } };
    }
  } finally {
    if (abort === mine) abort = null;
    stopTicker();
  }
  return true;
}

/** Cancel the running render (the server stops the job when the request is aborted). */
export function cancelRender(): void {
  abort?.abort();
  stopTicker();
  const s = renderState.value;
  if (s.status === "running") renderState.value = { ...s, status: s.outcome ? "done" : "idle" };
}

/** Forget everything (another session was opened). */
export function resetRender(): void {
  cancelRender();
  renderState.value = { ...EMPTY };
  showResult.value = false;
  holdCompare.value = false;
}
