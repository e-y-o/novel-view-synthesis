import { signal } from "@preact/signals";
import { getHealth, type Health } from "../api/client";

export type HealthState =
  { kind: "loading" } | { kind: "ok"; health: Health } | { kind: "error"; message: string };

/** Result of `GET /api/health`: proves the dev proxy (or the production server) is reachable. */
export const health = signal<HealthState>({ kind: "loading" });

export async function refreshHealth(): Promise<void> {
  health.value = { kind: "loading" };
  try {
    health.value = { kind: "ok", health: await getHealth() };
  } catch (e) {
    health.value = { kind: "error", message: e instanceof Error ? e.message : String(e) };
  }
}
