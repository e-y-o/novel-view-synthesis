import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, getHealth } from "../src/api/client";
import { connectEvents, parseEvent, wsUrl } from "../src/api/events";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const respond = (status: number, body: unknown) =>
  vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }));

describe("api client", () => {
  it("returns parsed JSON", async () => {
    vi.stubGlobal("fetch", respond(200, { ok: true, version: "1", stub: true }));
    expect(await getHealth()).toEqual({ ok: true, version: "1", stub: true });
  });

  it("turns the server error shape into ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      respond(409, { error: { code: "not_ready", message: "still analysing" } }),
    );
    const err = await api("/x").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 409, code: "not_ready", message: "still analysing" });
  });

  it("reports network failures with status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await expect(api("/x")).rejects.toMatchObject({ status: 0, code: "network" });
  });

  it("copes with non-JSON error bodies", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("boom", { status: 502 })));
    await expect(api("/x")).rejects.toMatchObject({ status: 502, code: "http_502" });
  });
});

describe("events", () => {
  it("builds ws and wss urls", () => {
    expect(wsUrl("/api/events", { protocol: "http:", host: "localhost:5173" })).toBe(
      "ws://localhost:5173/api/events",
    );
    expect(wsUrl("/a", { protocol: "https:", host: "h" })).toBe("wss://h/a");
  });

  it("parses only typed JSON text frames", () => {
    expect(parseEvent('{"type":"ready","session":"s"}')).toEqual({ type: "ready", session: "s" });
    expect(parseEvent("nope")).toBeNull();
    expect(parseEvent('{"x":1}')).toBeNull();
    expect(parseEvent(new ArrayBuffer(1))).toBeNull();
  });

  it("delivers events and reconnects with backoff until closed", () => {
    vi.useFakeTimers();
    vi.stubGlobal("location", { protocol: "http:", host: "h" });
    const sockets: WebSocket[] = [];
    const events: unknown[] = [];
    const conn = connectEvents("/api/events", {
      onEvent: (e) => events.push(e),
      backoffMs: [100, 200],
      socketFactory: (url) => {
        const s = { url, close: vi.fn() } as unknown as WebSocket;
        sockets.push(s);
        return s;
      },
    });
    expect(sockets).toHaveLength(1);
    const first = sockets[0] as WebSocket;
    first.onmessage?.({ data: '{"type":"ready","session":"a"}' } as MessageEvent);
    expect(events).toEqual([{ type: "ready", session: "a" }]);

    first.onclose?.({} as CloseEvent); // the server went away
    vi.advanceTimersByTime(99);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);

    (sockets[1] as WebSocket).onclose?.({} as CloseEvent); // second failure: 200 ms
    vi.advanceTimersByTime(200);
    expect(sockets).toHaveLength(3);

    conn.close();
    (sockets[2] as WebSocket).onclose?.({} as CloseEvent);
    vi.advanceTimersByTime(10_000);
    expect(sockets).toHaveLength(3); // no reconnect after close()
  });
});
