"""Job scheduler: one GPU job at a time, by priority, with coalescing, cancel, timeouts and crash retry.

The worker (``nvs.worker``) runs a single job at a time, so everything that decides *which* job goes next
lives here. Priority classes: ``INTERACTIVE`` (coverage, limit) > ``RENDER`` > ``ANALYZE``; within a class,
first come first served. Everything runs on the asyncio loop of the web process.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import statistics
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from nvs.worker.client import WorkerClient
from nvs.worker.protocol import ERROR, READY, RESULT, Event, Request

log = logging.getLogger("nvs.jobs")

WORKER_WAIT_S = 30.0  # how long a job may wait for the worker to become ready
CANCEL_GRACE_S = 2.0  # after a timeout: cooperative cancel first, then restart the worker
MAX_RETRIES = 1  # re-submissions after a worker crash
HISTORY = 100


class Priority(IntEnum):
    INTERACTIVE = 0
    RENDER = 1
    ANALYZE = 2


TIMEOUTS_S = {Priority.INTERACTIVE: 5.0, Priority.RENDER: 60.0, Priority.ANALYZE: 120.0}


@dataclass(frozen=True)
class JobOutcome:
    """How a job ended: ``result`` | ``error`` | ``cancelled`` | ``superseded`` | ``timeout``."""

    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None  # {"code", "message"} when status == "error" / "timeout"

    @property
    def ok(self) -> bool:
        return self.status == RESULT


class JobHandle:
    def __init__(self, scheduler: Scheduler, kind: str, payload: dict, priority: Priority, **opts: Any):
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.payload = payload
        self.priority = Priority(priority)
        self.session: str | None = opts.get("session")
        self.coalesce_key: str | None = opts.get("coalesce_key")
        self.timeout: float = opts.get("timeout") or TIMEOUTS_S[self.priority]
        self.on_event: Callable[[Event], None] | None = opts.get("on_event")
        self.state = "queued"  # queued | running | done
        self.attempts = 0
        self.cancel_requested = False
        self._scheduler = scheduler
        self._future: asyncio.Future[JobOutcome] = asyncio.get_running_loop().create_future()
        self._seq = 0
        self._enqueued = time.perf_counter()
        self._first_start: float | None = None
        self._request_id = ""

    @property
    def done(self) -> bool:
        return self._future.done()

    async def result(self) -> JobOutcome:
        """Wait for the outcome. Cancelling the waiter does not cancel the job (call ``cancel()``)."""
        return await asyncio.shield(self._future)

    def cancel(self) -> None:
        self._scheduler._cancel(self)


class Scheduler:
    def __init__(self, worker: WorkerClient, worker_wait_s: float = WORKER_WAIT_S):
        self.worker = worker
        self.worker_wait_s = worker_wait_s
        self._queue: list[JobHandle] = []
        self._seq = itertools.count()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._running: JobHandle | None = None
        self._closed = False
        self._waits: dict[Priority, deque[float]] = {p: deque(maxlen=HISTORY) for p in Priority}
        self._runs: dict[Priority, deque[float]] = {p: deque(maxlen=HISTORY) for p in Priority}

    # ------------------------------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="scheduler")

    async def close(self) -> None:
        """Stop scheduling; queued and running jobs resolve as ``cancelled``."""
        self._closed = True
        running = self._running  # the loop's ``finally`` clears it when the task is cancelled
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for job in [*self._queue, *([running] if running else [])]:
            if job._request_id and job.state == "running":
                self.worker.cancel(job._request_id)
            self._finish(job, JobOutcome("cancelled"))
        self._queue.clear()
        self._running = None

    # ---------------------------------------------------------------------------------------- submit
    def submit(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: Priority = Priority.RENDER,
        session: str | None = None,
        coalesce_key: str | None = None,
        timeout: float | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> JobHandle:
        job = JobHandle(
            self,
            kind,
            payload or {},
            priority,
            session=session,
            coalesce_key=coalesce_key,
            timeout=timeout,
            on_event=on_event,
        )
        if self._closed:
            self._finish(job, JobOutcome("cancelled"))
            return job
        if coalesce_key is not None:
            for old in [j for j in self._queue if j.coalesce_key == coalesce_key]:
                self._queue.remove(old)  # only queued jobs: a running one is never replaced
                self._finish(old, JobOutcome("superseded"))
        job._seq = next(self._seq)
        self._queue.append(job)
        self._wakeup.set()
        return job

    def _cancel(self, job: JobHandle) -> None:
        if job.done:
            return
        job.cancel_requested = True
        if job.state == "queued":
            if job in self._queue:
                self._queue.remove(job)
            self._finish(job, JobOutcome("cancelled"))
        elif job.state == "running":
            self.worker.cancel(job._request_id)  # the outcome arrives from the worker's `cancelled` event

    def _finish(self, job: JobHandle, outcome: JobOutcome) -> None:
        job.state = "done"
        if not job._future.done():
            job._future.set_result(outcome)

    # ------------------------------------------------------------------------------------ main loop
    async def _loop(self) -> None:
        while True:
            if not self._queue:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            if self.worker.status.state != READY:
                await self._wait_for_worker()
                continue
            job = min(self._queue, key=lambda j: (j.priority, j._seq))
            self._queue.remove(job)
            self._running = job
            try:
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # a scheduler bug must not wedge the queue
                log.exception("job %s (%s) crashed the scheduler", job.id, job.kind)
                self._finish(job, JobOutcome("error", error={"code": "internal", "message": str(e)}))
            finally:
                self._running = None

    async def _wait_for_worker(self) -> None:
        """Block until the worker is ready; fail jobs that have waited ``worker_wait_s`` for it."""
        now = time.perf_counter()
        expired = [j for j in self._queue if now - j._enqueued >= self.worker_wait_s]
        for job in expired:
            self._queue.remove(job)
            self._finish(
                job,
                JobOutcome(
                    "error",
                    error={
                        "code": "worker_unavailable",
                        "message": f"the GPU worker is not ready ({self.worker.status.detail or 'starting'})",
                    },
                ),
            )
        if not self._queue:
            return
        remaining = min(j._enqueued for j in self._queue) + self.worker_wait_s - now
        try:
            await self.worker.wait_ready(max(remaining, 0.01))
        except TimeoutError:
            pass

    async def _run(self, job: JobHandle) -> None:
        job.state = "running"
        job.attempts += 1
        started = time.perf_counter()
        if job._first_start is None:
            job._first_start = started
        job._request_id = f"{job.id}.{job.attempts}"
        request = Request(
            id=job._request_id,
            kind=job.kind,
            payload=job.payload,
            priority=int(job.priority),
            coalesce_key=job.coalesce_key,
        )
        handle = self.worker.submit(request)
        timed_out = False
        try:
            terminal = await asyncio.wait_for(self._consume(job, handle), job.timeout)
        except TimeoutError:
            timed_out = True
            terminal = Event(request.id, "timeout")
            self.worker.cancel(request.id)
            try:
                await asyncio.wait_for(handle.wait(), CANCEL_GRACE_S)
            except TimeoutError:
                log.warning("job %s ignored cancel after timeout; restarting the worker", job.id)
                await self.worker.restart()

        run_s = time.perf_counter() - started
        if (
            terminal.type == ERROR
            and terminal.data.get("code") == "worker_died"
            and not job.cancel_requested
            and job.attempts <= MAX_RETRIES
        ):
            log.warning("worker died during %s; retrying once after restart", job.kind)
            job.state = "queued"
            job._enqueued = time.perf_counter()
            self._queue.append(job)  # keeps its sequence number: still first in its class
            return
        self._waits[job.priority].append((job._first_start or started) - job._enqueued)
        self._runs[job.priority].append(run_s)
        if timed_out:
            outcome = JobOutcome(
                "timeout",
                error={"code": "timeout", "message": f"{job.kind} took longer than {job.timeout:g} s"},
            )
        elif terminal.type == RESULT:
            outcome = JobOutcome("result", terminal.data)
        elif terminal.type == ERROR:
            outcome = JobOutcome("error", error=dict(terminal.data))
        else:
            outcome = JobOutcome("cancelled")
        self._finish(job, outcome)

    async def _consume(self, job: JobHandle, handle: Any) -> Event:
        async for event in handle:
            if event.terminal:
                return event
            if job.on_event is not None:
                try:
                    job.on_event(event)
                except Exception:
                    log.exception("on_event callback failed")
        raise RuntimeError("event stream ended without a terminal event")

    # ---------------------------------------------------------------------------------------- stats
    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def running(self) -> JobHandle | None:
        return self._running

    def timings(self) -> dict[str, dict[str, float | int]]:
        """Median and p95 of queue wait and run time (ms) over the last 100 jobs per priority class."""

        def stats(values: deque[float]) -> tuple[float, float]:
            if not values:
                return 0.0, 0.0
            ordered = sorted(values)
            p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
            return statistics.median(ordered) * 1000, p95 * 1000

        out: dict[str, dict[str, float | int]] = {}
        for p in Priority:
            wait_med, wait_p95 = stats(self._waits[p])
            run_med, run_p95 = stats(self._runs[p])
            out[p.name.lower()] = {
                "count": len(self._runs[p]),
                "wait_median_ms": round(wait_med, 2),
                "wait_p95_ms": round(wait_p95, 2),
                "run_median_ms": round(run_med, 2),
                "run_p95_ms": round(run_p95, 2),
            }
        return out

    def status(self) -> dict[str, Any]:
        ws = self.worker.status
        running = self._running
        return {
            "worker": {"state": ws.state, "detail": ws.detail},
            "queue": {
                "length": len(self._queue),
                "running": None if running is None else {"id": running.id, "kind": running.kind},
            },
            "timings": self.timings(),
        }
