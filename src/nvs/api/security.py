"""Pure-ASGI guards for a loopback server: Host check, Origin check, security headers, request log.

Why: any web page open in the same browser can send requests to ``localhost`` (CSRF) or, through DNS
rebinding, read the responses. The server runs GPU jobs and writes files, so requests must come from its own
UI. There is deliberately no CORS middleware: cross-origin pages cannot read responses, and the Origin check
stops them from driving the server with simple POSTs or WebSockets.
"""

from __future__ import annotations

import json
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nvs.api.config import ServerConfig

log = logging.getLogger("nvs.api")

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CSP = (
    "default-src 'self'; img-src 'self' blob: data:; "
    "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; style-src 'self' 'unsafe-inline'"
)


def _header(scope: Scope, name: bytes) -> str | None:
    for k, v in scope["headers"]:
        if k == name:
            return v.decode("latin-1")
    return None


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, config: ServerConfig):
        self.app = app
        self.hosts = frozenset(h.lower() for h in config.allowed_hosts)
        self.origins = frozenset(o.lower() for o in config.allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = 0
        is_ws = scope["type"] == "websocket"

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = self._secure_headers(scope, message)
            elif message["type"] == "websocket.close" and not status:
                status = 403
            elif message["type"] == "websocket.accept" and not status:
                status = 101
            await send(message)

        rejection = self._check(scope)
        try:
            if rejection is not None:
                code, http_status, text = rejection
                await self._reject(scope, send_wrapper, code, http_status, text)
            else:
                await self.app(scope, receive, send_wrapper)
        finally:
            ms = (time.perf_counter() - started) * 1000
            method = "WS" if is_ws else scope["method"]
            # Path only: never the query string, headers or body.
            log.info("%s %s -> %s (%.1f ms)", method, scope["path"], status or "-", ms)

    def _check(self, scope: Scope) -> tuple[str, int, str] | None:
        host = (_header(scope, b"host") or "").lower()
        if host not in self.hosts:
            return "bad_host", 400, "Host header not allowed"
        if scope["type"] == "websocket" or scope["method"] in UNSAFE_METHODS:
            origin = _header(scope, b"origin")
            if origin is not None and origin.lower() not in self.origins:
                return "bad_origin", 403, "Origin not allowed"
        return None

    @staticmethod
    async def _reject(scope: Scope, send: Send, code: str, http_status: int, text: str) -> None:
        if scope["type"] == "websocket":
            # Closing before accept makes the server answer the handshake with HTTP 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": {"code": code, "message": text}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": http_status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _secure_headers(scope: Scope, message: Message) -> list[tuple[bytes, bytes]]:
        headers = list(message.get("headers", []))
        present = {k.lower() for k, _ in headers}
        content_type = next((v for k, v in headers if k.lower() == b"content-type"), b"")

        def add(name: bytes, value: str) -> None:
            if name not in present:
                headers.append((name, value.encode()))

        add(b"x-content-type-options", "nosniff")
        add(b"referrer-policy", "no-referrer")
        if scope["path"].startswith("/api/"):
            add(b"cache-control", "no-store")
        if content_type.startswith(b"text/html"):
            add(b"content-security-policy", CSP)
        return headers
