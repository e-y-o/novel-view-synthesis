/** Typed session endpoints (`/api/sessions`). */
import { ApiError, api } from "./client";
import type { PreviewMeta } from "../gl/assets";

export type SessionState = "uploaded" | "analyzing" | "ready" | "error";

export interface SessionSummary {
  id: string;
  filename: string;
  state: SessionState;
  created: number;
  last_used: number;
  error: { code: string; message: string } | null;
  width: number;
  height: number;
  bytes: number;
  /** Asset URL of `color.jpg` once the session is ready. */
  thumbnail: string | null;
}

export interface SessionDetail extends SessionSummary {
  meta: PreviewMeta | null;
}

export interface UploadResult {
  id: string;
  state: SessionState;
  filename: string;
  existing: boolean;
}

export const listSessions = async (): Promise<SessionSummary[]> =>
  (await api<{ sessions: SessionSummary[] }>("/api/sessions")).sessions;

export const getSession = (id: string): Promise<SessionDetail> =>
  api<SessionDetail>(`/api/sessions/${id}`);

export const deleteSession = (id: string): Promise<void> =>
  api<void>(`/api/sessions/${id}`, { method: "DELETE" });

export const assetUrl = (id: string, name: string): string => `/api/sessions/${id}/assets/${name}`;

/**
 * Upload a photo with progress (`fetch` cannot report upload progress, XMLHttpRequest can).
 * `onProgress` gets 0..100. Rejects with `ApiError` (status 0, code `network` or `aborted` on transport errors).
 */
export function uploadSession(
  file: File,
  onProgress: (pct: number) => void,
  signal?: AbortSignal,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/sessions");
    xhr.responseType = "text";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onProgress((100 * e.loaded) / e.total);
    };
    xhr.onerror = () => reject(new ApiError(0, "network", "could not reach the server"));
    xhr.onabort = () => reject(new ApiError(0, "aborted", "upload cancelled"));
    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = undefined;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(100);
        resolve(body as UploadResult);
        return;
      }
      const err = (body as { error?: { code?: string; message?: string } } | undefined)?.error;
      reject(
        new ApiError(
          xhr.status,
          err?.code ?? `http_${xhr.status}`,
          err?.message ?? (xhr.statusText || "upload failed"),
        ),
      );
    };
    signal?.addEventListener("abort", () => xhr.abort());
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}
