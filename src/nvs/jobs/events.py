"""Event hub: fan-out of JSON-able event dicts to WebSocket subscribers, with replay of the latest state.

Channels: one global channel (``session=None``, receives everything) and one per session (receives that
session's events plus worker ``status``). New subscribers first get the current state, so a page reload shows
the right thing: the last worker ``status`` and, for a session, its latest ``analyze.progress`` / ``ready`` /
``error``. Subscriber queues are bounded: when a slow client falls behind, the oldest ``analyze.progress``
events are dropped; ``ready``, ``error`` and ``status`` are never dropped, and nothing here ever blocks the
publisher.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

Event = dict[str, Any]
DROPPABLE = frozenset({"analyze.progress"})
REMEMBERED = frozenset({"analyze.progress", "ready", "error"})  # per session: only the latest of these
DEFAULT_MAX_QUEUE = 256


class Subscription:
    def __init__(self, hub: EventHub, session: str | None, max_queue: int):
        self._hub = hub
        self.session = session
        self._max = max_queue
        self._items: deque[Event] = deque()
        self._wake = asyncio.Event()
        self._closed = False
        self.dropped = 0

    def _push(self, event: Event) -> None:
        if self._closed:
            return
        if len(self._items) >= self._max:
            for i, queued in enumerate(self._items):  # make room by dropping the oldest droppable event
                if queued.get("type") in DROPPABLE:
                    del self._items[i]
                    self.dropped += 1
                    break
            else:
                if event.get("type") in DROPPABLE:
                    self.dropped += 1
                    return
        self._items.append(event)
        self._wake.set()

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        self._hub._remove(self)

    @property
    def pending(self) -> int:
        return len(self._items)

    async def get(self) -> Event | None:
        """Next event, or ``None`` once the subscription (or the hub) is closed and drained."""
        while True:
            if self._items:
                return self._items.popleft()
            if self._closed:
                return None
            self._wake.clear()
            await self._wake.wait()

    async def __aiter__(self) -> AsyncIterator[Event]:
        while (event := await self.get()) is not None:
            yield event

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EventHub:
    def __init__(self, max_queue: int = DEFAULT_MAX_QUEUE):
        self.max_queue = max_queue
        self._subs: set[Subscription] = set()
        self._status: Event | None = None
        self._state: dict[str, Event] = {}
        self._closed = False

    def publish(self, event: Event, session: str | None = None) -> None:
        """Deliver ``event`` (a dict with a ``type``). Call from the event loop thread."""
        if self._closed:
            return
        kind = event.get("type")
        if kind == "status":
            self._status = event
        elif session is not None and kind in REMEMBERED:
            self._state[session] = event
        for sub in list(self._subs):
            if sub.session is None or kind == "status" or sub.session == session:
                sub._push(event)

    def subscribe(self, session: str | None = None) -> Subscription:
        sub = Subscription(self, session, self.max_queue)
        if self._status is not None:
            sub._push(self._status)
        if session is not None and session in self._state:
            sub._push(self._state[session])
        if self._closed:
            sub.close()
        else:
            self._subs.add(sub)
        return sub

    def forget(self, session: str) -> None:
        self._state.pop(session, None)

    def _remove(self, sub: Subscription) -> None:
        self._subs.discard(sub)

    def close(self) -> None:
        """End every subscription (server shutdown); queued events are still delivered first."""
        self._closed = True
        for sub in list(self._subs):
            sub.close()

    @property
    def subscribers(self) -> int:
        return len(self._subs)
