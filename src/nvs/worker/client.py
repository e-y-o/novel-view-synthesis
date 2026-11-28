"""Server-side handle on the GPU worker process (asyncio friendly, torch-free).

The worker is a spawned child process connected by two one-way pipes. A reader thread turns its events into
asyncio callbacks. If the worker dies, every in-flight request fails with ``error{code: "worker_died"}`` and
the client restarts it with a backoff; after too many consecutive failures it gives up and reports ``error``.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from nvs.worker.main import worker_main
from nvs.worker.protocol import (
    ERROR,
    FAILED,
    READY,
    STARTING,
    Cancel,
    Event,
    Request,
    Shutdown,
    WorkerConfig,
    WorkerStatus,
)

log = logging.getLogger("nvs.worker.client")

DEFAULT_BACKOFF = (1.0, 5.0, 30.0)
STABLE_AFTER_S = 60.0  # a worker that stayed up this long resets the failure count


class JobHandle:
    """Events of one request. Iterate for all events, or ``await wait()`` for the terminal one."""

    def __init__(self, request: Request):
        self.request = request
        self.events: list[Event] = []
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._done = False

    def _push(self, event: Event) -> None:
        if self._done:
            return
        self._done = event.terminal
        self.events.append(event)
        self._queue.put_nowait(event)

    @property
    def done(self) -> bool:
        return self._done

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            event = await self._queue.get()
            yield event
            if event.terminal:
                return

    async def wait(self, timeout: float | None = None) -> Event:
        """Consume events until the terminal one and return it."""

        async def drain() -> Event:
            async for event in self:
                if event.terminal:
                    return event
            raise RuntimeError("unreachable")

        return await asyncio.wait_for(drain(), timeout)


class WorkerClient:
    def __init__(
        self,
        config: WorkerConfig,
        *,
        auto_restart: bool = True,
        backoff: tuple[float, ...] = DEFAULT_BACKOFF,
    ):
        self.config = config
        self.auto_restart = auto_restart
        self.backoff = backoff
        self._status = WorkerStatus(STARTING, "not started")
        self._listeners: list[Callable[[WorkerStatus], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._proc: Any = None
        self._send_conn: Any = None
        self._recv_conn: Any = None
        self._reader: threading.Thread | None = None
        self._gen = 0
        self._started_at = 0.0
        self._failures = 0
        self._stopping = False
        self._inflight: dict[str, JobHandle] = {}
        self._ready = asyncio.Event()
        self._restart_task: asyncio.Task | None = None

    # ------------------------------------------------------------------------------------------ state
    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def process(self) -> Any:
        return self._proc

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def add_status_listener(self, callback: Callable[[WorkerStatus], None]) -> None:
        self._listeners.append(callback)

    def _set_status(self, status: WorkerStatus) -> None:
        self._status = status
        if status.state == READY:
            self._ready.set()
        else:
            self._ready.clear()
        for cb in list(self._listeners):
            try:
                cb(status)
            except Exception:
                log.exception("status listener failed")

    async def wait_ready(self, timeout: float | None = 30.0) -> WorkerStatus:
        """Wait until the worker reports ``ready``; raises ``TimeoutError`` otherwise."""
        await asyncio.wait_for(self._ready.wait(), timeout)
        return self._status

    # ------------------------------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        await self._spawn()

    async def _spawn(self) -> None:
        loop = asyncio.get_running_loop()
        ctx = mp.get_context("spawn")
        req_recv, req_send = ctx.Pipe(duplex=False)
        ev_recv, ev_send = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=worker_main, args=(req_recv, ev_send, self.config), name="nvs-worker", daemon=True
        )
        self._gen += 1
        gen = self._gen
        self._proc, self._send_conn, self._recv_conn = proc, req_send, ev_recv
        self._started_at = time.monotonic()
        self._set_status(WorkerStatus(STARTING, "starting worker process"))
        await loop.run_in_executor(None, proc.start)
        # The child owns these ends now; closing ours is what makes EOF/BrokenPipe work both ways.
        req_recv.close()
        ev_send.close()
        self._reader = threading.Thread(
            target=self._read_loop, args=(ev_recv, gen), daemon=True, name=f"worker-reader-{gen}"
        )
        self._reader.start()

    async def stop(self, timeout: float = 5.0) -> None:
        """Shut the worker down (politely, then by force). No process is left behind."""
        self._stopping = True
        if self._restart_task is not None:
            self._restart_task.cancel()
            self._restart_task = None
        await self._terminate(timeout)
        self._fail_inflight("worker stopped", code="worker_stopped")

    async def restart(self) -> None:
        """Replace the worker (also after it was given up on). Resets the failure count."""
        if self._restart_task is not None:
            self._restart_task.cancel()
            self._restart_task = None
        self._stopping = True  # silence death handling for the old process
        await self._terminate(5.0)
        self._fail_inflight("worker restarted", code="worker_died")
        self._stopping = False
        self._failures = 0
        await self._spawn()

    async def _terminate(self, timeout: float) -> None:
        proc, conn, reader = self._proc, self._send_conn, self._reader
        loop = asyncio.get_running_loop()
        if proc is not None:
            if proc.is_alive():
                try:
                    conn.send(Shutdown())
                except (OSError, ValueError):
                    pass
                await loop.run_in_executor(None, proc.join, timeout)
                if proc.is_alive():
                    proc.kill()
                    await loop.run_in_executor(None, proc.join, 5.0)
            else:
                await loop.run_in_executor(None, proc.join, 1.0)
        for c in (conn, self._recv_conn):
            try:
                if c is not None:
                    c.close()
            except OSError:
                pass
        if reader is not None:
            await loop.run_in_executor(None, reader.join, 2.0)
        self._gen += 1  # events still in flight from the old process are stale
        self._proc = self._send_conn = self._recv_conn = self._reader = None
        self._set_status(WorkerStatus(FAILED, "worker stopped"))

    # ---------------------------------------------------------------------------------------- reading
    def _read_loop(self, conn: Any, gen: int) -> None:
        try:
            while True:
                obj = conn.recv()
                self._call(self._dispatch, gen, obj)
        except (EOFError, OSError, ValueError):
            pass
        self._call(self._on_dead, gen)

    def _call(self, fn: Callable, *args: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # loop closed during shutdown
            pass

    def _dispatch(self, gen: int, obj: Any) -> None:
        if gen != self._gen:
            return
        if isinstance(obj, WorkerStatus):
            self._set_status(obj)
        elif isinstance(obj, Event):
            handle = self._inflight.get(obj.request_id)
            if handle is not None:
                handle._push(obj)
                if obj.terminal:
                    self._inflight.pop(obj.request_id, None)

    def _fail_inflight(self, message: str, code: str = "worker_died") -> None:
        for rid, handle in list(self._inflight.items()):
            handle._push(Event(rid, ERROR, {"code": code, "message": message}))
        self._inflight.clear()

    def _on_dead(self, gen: int) -> None:
        if gen != self._gen or self._stopping:
            return
        proc = self._proc
        code = proc.exitcode if proc is not None else None
        message = f"the GPU worker process died (exit code {code})"
        log.error(message)
        self._fail_inflight(message)
        if time.monotonic() - self._started_at > STABLE_AFTER_S:
            self._failures = 0
        self._failures += 1
        if self.auto_restart and self._failures <= len(self.backoff):
            delay = self.backoff[self._failures - 1]
            self._set_status(WorkerStatus(FAILED, f"{message}; restarting in {delay:g} s"))
            self._restart_task = asyncio.ensure_future(self._restart_after(delay))
        else:
            self._set_status(WorkerStatus(FAILED, f"{message}; giving up after {self._failures} failures"))

    async def _restart_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        if self._stopping:
            return
        self._gen += 1  # old process is already dead; make sure nothing stale gets through
        for c in (self._send_conn, self._recv_conn):
            try:
                if c is not None:
                    c.close()
            except OSError:
                pass
        await self._spawn()

    # ------------------------------------------------------------------------------------------ jobs
    def submit(self, request: Request) -> JobHandle:
        """Queue a request. Never raises: problems arrive as an ``error`` event on the handle."""
        handle = JobHandle(request)
        if not self.alive or self._send_conn is None:
            handle._push(
                Event(request.id, ERROR, {"code": "worker_died", "message": "the GPU worker is not running"})
            )
            return handle
        self._inflight[request.id] = handle
        try:
            self._send_conn.send(request)
        except (OSError, ValueError) as e:
            self._inflight.pop(request.id, None)
            handle._push(Event(request.id, ERROR, {"code": "worker_died", "message": f"worker pipe: {e}"}))
        return handle

    def cancel(self, request_id: str) -> None:
        if self._send_conn is None:
            return
        try:
            self._send_conn.send(Cancel(request_id))
        except (OSError, ValueError):
            pass
