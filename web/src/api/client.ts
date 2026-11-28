/** Thin fetch wrapper for the Python server. Errors arrive as `{error: {code, message}}`. */

export interface Health {
  ok: boolean;
  version: string;
  stub: boolean;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface ErrorBody {
  error?: { code?: string; message?: string };
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (e) {
    throw new ApiError(0, "network", e instanceof Error ? e.message : "network error");
  }
  const text = await response.text();
  let body: unknown = undefined;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      if (response.ok) throw new ApiError(response.status, "bad_response", "response is not JSON");
    }
  }
  if (!response.ok) {
    const err = (body as ErrorBody | undefined)?.error;
    throw new ApiError(
      response.status,
      err?.code ?? `http_${response.status}`,
      err?.message ?? (response.statusText || "request failed"),
    );
  }
  return body as T;
}

export function postJson<T>(path: string, payload: unknown, init?: RequestInit): Promise<T> {
  return api<T>(path, {
    ...init,
    method: "POST",
    headers: { "Content-Type": "application/json", ...init?.headers },
    body: JSON.stringify(payload),
  });
}

export const getHealth = (): Promise<Health> => api<Health>("/api/health");
