"""WebSocket event streams and ``GET /api/status``.

Messages are JSON text frames ``{type, ...}`` (contract: ``docs/tickets/m2/README.md``). Keep-alive pings are
WebSocket protocol pings sent by uvicorn (``ws_ping_interval=20`` in ``nvs serve``), not messages. On server
shutdown the hub closes every subscription and the sockets are closed cleanly.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from nvs.api.sessions import SessionStore
from nvs.jobs.events import EventHub, Subscription
from nvs.jobs.scheduler import Scheduler

WS_NORMAL = 1000
WS_GOING_AWAY = 1001
WS_NOT_FOUND = 4404


async def _pump(ws: WebSocket, sub: Subscription) -> None:
    """Send hub events until the client leaves or the hub closes."""

    async def sender() -> None:
        async for event in sub:
            await ws.send_json(event)

    async def receiver() -> None:  # we expect no messages, but must notice the disconnect
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return

    tasks = [asyncio.ensure_future(sender()), asyncio.ensure_future(receiver())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    sender_finished = tasks[0] in done and not tasks[0].cancelled()
    if sender_finished and tasks[0].exception() is None:  # hub closed: server is shutting down
        try:
            await ws.close(code=WS_GOING_AWAY)
        except RuntimeError:
            pass


def make_events_router(hub: EventHub, store: SessionStore, scheduler: Scheduler) -> APIRouter:
    router = APIRouter()

    @router.websocket("/api/events")
    async def global_events(ws: WebSocket) -> None:
        await ws.accept()
        with hub.subscribe(None) as sub:
            try:
                await _pump(ws, sub)
            except (WebSocketDisconnect, RuntimeError):
                pass

    @router.websocket("/api/sessions/{session_id}/events")
    async def session_events(ws: WebSocket, session_id: str) -> None:
        if store.get(session_id) is None:
            await ws.close(code=WS_NOT_FOUND)  # before accept: the handshake is refused
            return
        await ws.accept()
        with hub.subscribe(session_id) as sub:
            try:
                await _pump(ws, sub)
            except (WebSocketDisconnect, RuntimeError):
                pass

    @router.get("/api/status")
    async def status() -> dict:
        return scheduler.status()

    return router
