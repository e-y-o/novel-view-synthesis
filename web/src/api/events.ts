/** WebSocket event stream (`/api/events`, `/api/sessions/<id>/events`) with automatic reconnect. */

export type ServerEvent =
  | { type: "status"; worker: "starting" | "ready" | "error"; detail: string }
  | { type: "analyze.progress"; session: string; phase: string; pct: number }
  | { type: "ready"; session: string }
  | { type: "error"; session?: string; code: string; message: string };

export interface EventsOptions {
  onEvent: (event: ServerEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
  /** Reconnect delays in ms; the last one repeats. */
  backoffMs?: readonly number[];
  /** Injectable for tests. */
  socketFactory?: (url: string) => WebSocket;
}

export interface EventsConnection {
  close(): void;
}

export function wsUrl(path: string, loc: Pick<Location, "protocol" | "host"> = location): string {
  return `${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}${path}`;
}

export function parseEvent(data: unknown): ServerEvent | null {
  if (typeof data !== "string") return null;
  try {
    const value: unknown = JSON.parse(data);
    if (
      typeof value === "object" &&
      value !== null &&
      typeof (value as { type?: unknown }).type === "string"
    ) {
      return value as ServerEvent;
    }
  } catch {
    /* ignore malformed frames */
  }
  return null;
}

export function connectEvents(path: string, options: EventsOptions): EventsConnection {
  const backoff = options.backoffMs ?? [500, 1000, 2000, 5000];
  const make = options.socketFactory ?? ((url: string) => new WebSocket(url));
  let socket: WebSocket | null = null;
  let attempt = 0;
  let closed = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const open = (): void => {
    socket = make(wsUrl(path));
    socket.onopen = () => {
      attempt = 0;
      options.onOpen?.();
    };
    socket.onmessage = (message) => {
      const event = parseEvent(message.data);
      if (event) options.onEvent(event);
    };
    socket.onclose = () => {
      options.onClose?.();
      if (closed) return;
      const delay = backoff[Math.min(attempt, backoff.length - 1)] ?? 1000;
      attempt += 1;
      timer = setTimeout(open, delay);
    };
  };
  open();

  return {
    close() {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
      socket?.close();
    },
  };
}
