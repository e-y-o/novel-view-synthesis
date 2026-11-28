/** Coverage and limit endpoints, and the latest-wins request loop that keeps them from piling up. */
import type { OrbitJson, PoseJson } from "../camera";
import { ApiError, postJson } from "./client";

export interface Coverage {
  disocclusion: number;
  stretch: number;
  out_of_frame: number;
  total: number;
}

export interface LimitResult {
  scale: number;
  yaw_deg: number;
  pitch_deg: number;
  dolly: number;
  pan: [number, number];
  hole_fraction: number;
  limited: boolean;
}

export interface LimitOptions {
  maxHoleFraction: number;
  countOutOfFrame: boolean;
}

export async function fetchCoverage(
  id: string,
  pose: PoseJson,
  signal?: AbortSignal,
): Promise<{ coverage: Coverage; ms: number }> {
  return postJson(`/api/sessions/${id}/coverage`, pose, { signal });
}

export async function fetchLimit(
  id: string,
  orbit: OrbitJson,
  options: LimitOptions,
  signal?: AbortSignal,
): Promise<LimitResult> {
  return postJson(
    `/api/sessions/${id}/limit`,
    {
      orbit,
      max_hole_fraction: options.maxHoleFraction,
      count_out_of_frame: options.countOutOfFrame,
    },
    { signal },
  );
}

/** Errors that mean "a newer request replaced this one" or "we cancelled it": never shown to the user. */
export function isBenign(e: unknown): boolean {
  if (e instanceof DOMException && e.name === "AbortError") return true;
  if (e instanceof ApiError) return e.code === "superseded" || e.code === "cancelled";
  return false;
}

export interface LoopOptions<P, R> {
  send: (payload: P, signal: AbortSignal) => Promise<R>;
  /** Called for every result that is for the newest payload (never for stale ones). */
  onResult: (result: R, payload: P) => void;
  /** True while the newest payload has no result yet (dim the old value). */
  onStale?: (stale: boolean) => void;
  onError?: (error: unknown) => void;
  /** Schedules `cb` for the next animation frame; injectable for tests. */
  frame?: (cb: () => void) => number;
  cancelFrame?: (id: number) => void;
}

/**
 * Latest-wins request loop: `update(payload)` may be called as often as you like; at most one request is in
 * flight, the next one is sent as soon as the previous returns (not before the next animation frame when idle),
 * and it always carries the newest payload. Results for payloads that were replaced meanwhile are dropped.
 */
export class LatestWinsLoop<P, R> {
  private pending: { payload: P } | null = null;
  private inFlight = false;
  private frameId = 0;
  private abort: AbortController | null = null;
  private disposed = false;
  private readonly frame: (cb: () => void) => number;
  private readonly cancelFrame: (id: number) => void;
  sent = 0;

  constructor(private readonly o: LoopOptions<P, R>) {
    this.frame = o.frame ?? ((cb) => requestAnimationFrame(cb));
    this.cancelFrame = o.cancelFrame ?? ((id) => cancelAnimationFrame(id));
  }

  update(payload: P): void {
    if (this.disposed) return;
    this.pending = { payload };
    this.o.onStale?.(true);
    if (!this.inFlight && !this.frameId) {
      this.frameId = this.frame(() => {
        this.frameId = 0;
        this.flush();
      });
    }
  }

  private flush(): void {
    if (this.disposed || this.inFlight || !this.pending) return;
    const { payload } = this.pending;
    this.pending = null;
    this.inFlight = true;
    this.abort = new AbortController();
    const signal = this.abort.signal;
    this.sent += 1;
    this.o
      .send(payload, signal)
      .then(
        (result) => {
          if (this.disposed || signal.aborted) return;
          if (!this.pending) {
            this.o.onResult(result, payload);
            this.o.onStale?.(false);
          }
          // else: the pose moved on while this was in flight; the newer one is sent right below
        },
        (error: unknown) => {
          if (!this.disposed && !isBenign(error)) this.o.onError?.(error);
        },
      )
      .finally(() => {
        this.inFlight = false;
        this.abort = null;
        if (this.pending) this.flush(); // newest pose immediately, no frame wait
      });
  }

  /** Cancel everything pending and in flight; later updates are ignored. */
  dispose(): void {
    this.disposed = true;
    this.pending = null;
    if (this.frameId) this.cancelFrame(this.frameId);
    this.frameId = 0;
    this.abort?.abort();
  }
}
