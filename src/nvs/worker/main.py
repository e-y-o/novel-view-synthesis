"""GPU worker process: owns the depth backend and runs jobs one at a time.

``worker_main`` is the ``multiprocessing`` (spawn) entry point. Torch is only imported inside it, so importing
this module (for the ``HANDLERS`` registry or by the web process) stays cheap.

Threads inside the worker: a reader thread receives control messages (so ``Cancel`` works while a job runs)
and feeds a queue; the main thread runs jobs. Cancellation is cooperative: handlers call
``ctx.check_cancel()`` or wait on ``ctx.sleep``.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from nvs.worker.protocol import (
    CANCELLED,
    ERROR,
    FAILED,
    PROGRESS,
    READY,
    RESULT,
    STARTED,
    STARTING,
    Cancel,
    Event,
    Request,
    Shutdown,
    WorkerConfig,
    WorkerStatus,
)

log = logging.getLogger("nvs.worker")

_EOF = object()  # the server closed the pipe (it exited): shut down


class JobCancelled(Exception):
    """Raised inside a handler when its request was cancelled."""


class JobError(Exception):
    """A failure with a stable ``code`` and a user-readable message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerState:
    """What handlers may use: the backend (lazily loaded) and the config."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.backend: Any = None
        self.load_error: str | None = None

    def make_backend(self) -> Any:
        if self.config.stub:
            from nvs.depth.stub import StubDepthBackend

            return StubDepthBackend()
        from nvs.depth.depthpro2 import DepthPro2Backend

        c = self.config
        return DepthPro2Backend(
            model=c.depth_model, precision=c.precision, max_mp=c.max_mp, vram_fraction=c.vram_fraction
        )

    def load(self) -> str:
        """Create and load the backend; returns a one-line description for the status detail."""
        t0 = time.perf_counter()
        backend = self.make_backend()
        backend.load()
        self.backend = backend
        dt = time.perf_counter() - t0
        detail = f"{backend.name} ready in {dt:.1f} s"
        if not self.config.stub:
            import torch

            vram = torch.cuda.memory_allocated() / 2**30
            reserved = torch.cuda.memory_reserved() / 2**30
            detail += f", {vram:.2f} GiB allocated / {reserved:.2f} GiB reserved VRAM"
        return detail

    def require_backend(self) -> Any:
        if self.backend is None:
            try:
                self.load()
            except Exception as e:
                self.load_error = f"{type(e).__name__}: {e}"
                raise JobError("backend_unavailable", f"depth backend could not load: {e}") from e
        return self.backend


class JobContext:
    def __init__(self, request: Request, state: WorkerState, cancel: threading.Event, send: Callable):
        self.request = request
        self.state = state
        self._cancel = cancel
        self._send = send

    def progress(self, pct: float, **data: Any) -> None:
        self._send(Event(self.request.id, PROGRESS, {"pct": float(pct), **data}))

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled

    def sleep(self, seconds: float) -> None:
        """Sleep, but wake immediately (raising ``JobCancelled``) when cancelled."""
        if self._cancel.wait(seconds):
            raise JobCancelled

    def backend(self) -> Any:
        return self.state.require_backend()


Handler = Callable[[JobContext, dict[str, Any]], dict[str, Any]]
HANDLERS: dict[str, Handler] = {}


def handler(kind: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return register


@handler("ping")
def _ping(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    return {"pong": True, "pid": os.getpid(), "stub": ctx.state.config.stub}


@handler("echo_sleep")
def _echo_sleep(ctx: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Test job: sleep ``seconds`` in ``steps`` slices, reporting progress, then echo ``payload``."""
    seconds = float(payload.get("seconds", 0.0))
    steps = max(1, int(payload.get("steps", 4)))
    for i in range(steps):
        ctx.sleep(seconds / steps)
        ctx.progress(100.0 * (i + 1) / steps, step=i + 1)
    return {"echo": payload.get("echo")}


# --------------------------------------------------------------------------------------------- main loop
def _reader(conn: Any, inbox: queue.Queue, cancel_flags: dict[str, threading.Event], lock: threading.Lock):
    """Receive messages: ``Cancel`` sets the flag at once, even while a job runs; the rest is queued."""
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            inbox.put(_EOF)
            return
        if isinstance(msg, Request):
            with lock:
                cancel_flags[msg.id] = threading.Event()
            inbox.put(msg)
        elif isinstance(msg, Cancel):
            with lock:
                flag = cancel_flags.get(msg.request_id)
            if flag is not None:
                flag.set()
        elif isinstance(msg, Shutdown):
            inbox.put(msg)
            return


def worker_main(conn_requests: Any, conn_events: Any, config: WorkerConfig) -> None:
    # Allocator options must be set before CUDA initialises (same as depth.base.configure_cuda_env, done by
    # hand so that nothing torch-related is imported first).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    send_lock = threading.Lock()

    def send(obj: Any) -> None:
        with send_lock:
            conn_events.send(obj)

    state = WorkerState(config)
    inbox: queue.Queue = queue.Queue()
    cancel_flags: dict[str, threading.Event] = {}
    flags_lock = threading.Lock()
    threading.Thread(
        target=_reader, args=(conn_requests, inbox, cancel_flags, flags_lock), daemon=True, name="reader"
    ).start()

    try:
        send(WorkerStatus(STARTING, "loading" if config.preload_depth and not config.stub else ""))
        if config.preload_depth:
            try:
                detail = state.load()
                log.info(detail)
                send(WorkerStatus(READY, detail))
            except Exception as e:
                state.load_error = f"{type(e).__name__}: {e}"
                log.error("depth backend failed to load: %s", state.load_error)
                send(WorkerStatus(FAILED, f"depth model could not be loaded: {e}"))
        else:
            send(WorkerStatus(READY, "depth model not preloaded"))

        while True:
            item = inbox.get()
            if item is _EOF or isinstance(item, Shutdown):
                break
            _run_job(item, state, cancel_flags, flags_lock, send)
    except (BrokenPipeError, OSError):
        pass  # server gone
    finally:
        if state.backend is not None:
            try:
                state.backend.unload()
            except Exception:  # shutting down anyway
                log.exception("unload failed")


def _run_job(
    request: Request,
    state: WorkerState,
    cancel_flags: dict[str, threading.Event],
    lock: threading.Lock,
    send: Callable,
) -> None:
    with lock:
        cancel = cancel_flags[request.id]
    try:
        if cancel.is_set():  # cancelled while queued
            send(Event(request.id, CANCELLED))
            return
        fn = HANDLERS.get(request.kind)
        if fn is None:
            send(
                Event(request.id, ERROR, {"code": "unknown_job", "message": f"unknown job {request.kind!r}"})
            )
            return
        send(Event(request.id, STARTED))
        try:
            result = fn(JobContext(request, state, cancel, send), request.payload)
        except JobCancelled:
            send(Event(request.id, CANCELLED))
        except JobError as e:
            send(Event(request.id, ERROR, {"code": e.code, "message": e.message}))
        except Exception as e:
            log.error("job %s (%s) failed:\n%s", request.id, request.kind, traceback.format_exc())
            send(Event(request.id, ERROR, {"code": "internal", "message": f"{type(e).__name__}: {e}"}))
        else:
            send(Event(request.id, RESULT, result if result is not None else {}))
    finally:
        with lock:
            cancel_flags.pop(request.id, None)
