"""Scheduler, event hub and WebSocket routes (T04). A real stub worker process runs ``echo_sleep`` jobs."""

import asyncio
import io
import statistics
import time

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from nvs.api.app import create_app
from nvs.api.config import ServerConfig
from nvs.jobs.events import EventHub
from nvs.jobs.scheduler import Priority, Scheduler
from nvs.worker.client import WorkerClient
from nvs.worker.protocol import WorkerConfig

STUB = WorkerConfig(stub=True)
PORT = 8765


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 90))


class running_scheduler:
    def __init__(self, **worker_kw):
        self.worker = WorkerClient(STUB, **worker_kw)
        self.scheduler = Scheduler(self.worker)

    async def __aenter__(self) -> Scheduler:
        await self.worker.start()
        await self.worker.wait_ready(30)
        self.scheduler.start()
        return self.scheduler

    async def __aexit__(self, *exc):
        await self.scheduler.close()
        await self.worker.stop()
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        assert pending == [], f"tasks left pending: {pending}"


def sleep_job(echo, seconds=0.1):
    return {"seconds": seconds, "steps": 2, "echo": echo}


def test_priority_order_coalescing_cancel_timeout_overhead():
    async def scenario():
        async with running_scheduler() as s:
            # ---- priority: three jobs wait behind a running one; they run interactive, render, analyze
            order: list[str] = []

            def track(name):
                return lambda e: order.append(name) if e.type == "started" else None

            blocker = s.submit("echo_sleep", sleep_job("blocker", 0.3), priority=Priority.RENDER)
            await asyncio.sleep(0.1)
            assert blocker.state == "running"
            a = s.submit("echo_sleep", sleep_job("a"), priority=Priority.ANALYZE, on_event=track("analyze"))
            r = s.submit("echo_sleep", sleep_job("r"), priority=Priority.RENDER, on_event=track("render"))
            i = s.submit("echo_sleep", sleep_job("i"), priority=Priority.INTERACTIVE, on_event=track("inter"))
            assert s.queue_length == 3
            for h in (blocker, a, r, i):
                assert (await h.result()).ok
            assert order == ["inter", "render", "analyze"]

            # ---- FIFO inside a class
            order.clear()
            blocker = s.submit("echo_sleep", sleep_job("blocker", 0.2))
            await asyncio.sleep(0.05)
            hs = [s.submit("echo_sleep", sleep_job(n, 0.01), on_event=track(n)) for n in "xyz"]
            await asyncio.gather(*(h.result() for h in [blocker, *hs]))
            assert order == ["x", "y", "z"]

            # ---- coalescing: ten queued jobs with the same key -> one runs, nine are superseded
            blocker = s.submit("echo_sleep", sleep_job("blocker", 0.3))
            await asyncio.sleep(0.1)
            cov = [
                s.submit(
                    "echo_sleep",
                    sleep_job(n, 0.01),
                    priority=Priority.INTERACTIVE,
                    coalesce_key="coverage:s1",
                )
                for n in range(10)
            ]
            outcomes = [await h.result() for h in cov]
            assert [o.status for o in outcomes] == ["superseded"] * 9 + ["result"]
            assert outcomes[-1].data["echo"] == 9  # the latest one ran
            assert (await blocker.result()).ok

            # a running job is never replaced by coalescing
            running = s.submit("echo_sleep", sleep_job("old", 0.4), coalesce_key="k")
            await asyncio.sleep(0.15)
            assert running.state == "running"
            newer = s.submit("echo_sleep", sleep_job("new", 0.01), coalesce_key="k")
            assert (await running.result()).status == "result"
            assert (await newer.result()).data["echo"] == "new"

            # ---- cancel queued
            blocker = s.submit("echo_sleep", sleep_job("blocker", 0.3))
            await asyncio.sleep(0.1)
            queued = s.submit("echo_sleep", sleep_job("q", 0.1))
            queued.cancel()
            assert (await queued.result()).status == "cancelled" and s.queue_length == 0
            assert (await blocker.result()).ok

            # ---- cancel running
            long = s.submit("echo_sleep", {"seconds": 10.0, "steps": 100})
            await asyncio.sleep(0.3)
            assert long.state == "running"
            t = time.perf_counter()
            long.cancel()
            assert (await long.result()).status == "cancelled"
            assert time.perf_counter() - t < 1.0
            long.cancel()  # idempotent after completion

            # worker still usable afterwards
            assert (await s.submit("ping").result()).data["pong"] is True

            # ---- timeout
            slow = s.submit("echo_sleep", {"seconds": 5.0, "steps": 50}, timeout=0.3)
            out = await slow.result()
            assert out.status == "timeout" and out.error["code"] == "timeout"
            assert (await s.submit("ping").result()).ok  # cancelled cooperatively, worker fine
            assert s.worker.status.state == "ready"

            # ---- unknown job kind surfaces as error
            bad = await s.submit("nope").result()
            assert bad.status == "error" and bad.error["code"] == "unknown_job"

            # ---- overhead of a no-op job and metrics
            await s.submit("ping").result()
            times = []
            for _ in range(50):
                t = time.perf_counter()
                assert (await s.submit("ping", priority=Priority.INTERACTIVE).result()).ok
                times.append((time.perf_counter() - t) * 1000)
            median = statistics.median(times)
            print(f"submit -> result, no-op job: median {median:.2f} ms, max {max(times):.2f} ms")
            assert median < 5.0
            status = s.status()
            assert status["worker"]["state"] == "ready"
            assert status["queue"] == {"length": 0, "running": None}
            inter = status["timings"]["interactive"]
            assert (
                inter["count"] >= 50
                and inter["run_median_ms"] > 0
                and inter["run_p95_ms"] >= inter["run_median_ms"]
            )

    run(scenario())


def test_worker_crash_job_is_retried_once():
    async def scenario():
        async with running_scheduler(backoff=(0.1, 0.1, 0.1)) as s:
            queued_behind = None
            job = s.submit("echo_sleep", {"seconds": 1.0, "steps": 10, "echo": "x"})
            await asyncio.sleep(0.4)
            queued_behind = s.submit("ping")
            s.worker.process.kill()
            out = await asyncio.wait_for(job.result(), 60)
            assert out.ok and out.data["echo"] == "x" and job.attempts == 2  # died once, retried, finished
            assert (await queued_behind.result()).ok  # queued jobs are kept across the restart

            # dies on the retry too: reported as worker_died, no endless loop
            job = s.submit("echo_sleep", {"seconds": 1.5, "steps": 15})
            for attempt in (1, 2):
                while job.attempts < attempt or job.state != "running":
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.3)
                s.worker.process.kill()
                await asyncio.sleep(0.2)
            out = await asyncio.wait_for(job.result(), 60)
            assert out.status == "error" and out.error["code"] == "worker_died" and job.attempts == 2
            await s.worker.wait_ready(30)
            assert (await s.submit("ping").result()).ok

    run(scenario())


def test_jobs_wait_for_worker_then_fail_friendly():
    async def scenario():
        worker = WorkerClient(STUB, auto_restart=False)  # never started: stays "starting"
        s = Scheduler(worker, worker_wait_s=0.3)
        s.start()
        try:
            t = time.perf_counter()
            out = await s.submit("ping").result()
            assert out.status == "error" and out.error["code"] == "worker_unavailable"
            assert 0.25 < time.perf_counter() - t < 2.0
        finally:
            await s.close()

    run(scenario())


def test_jobs_submitted_before_ready_run_when_ready():
    async def scenario():
        worker = WorkerClient(STUB)
        s = Scheduler(worker)
        s.start()
        job = s.submit("ping")  # worker not even started yet
        await asyncio.sleep(0.1)
        await worker.start()
        try:
            assert (await asyncio.wait_for(job.result(), 30)).ok
        finally:
            await s.close()
            await worker.stop()

    run(scenario())


def test_close_resolves_outstanding_jobs():
    async def scenario():
        async with running_scheduler() as s:
            running = s.submit("echo_sleep", {"seconds": 10.0, "steps": 100})
            queued = s.submit("ping")
            await asyncio.sleep(0.3)
            await s.close()
            assert (await running.result()).status == "cancelled"
            assert (await queued.result()).status == "cancelled"
            assert (await s.submit("ping").result()).status == "cancelled"  # closed scheduler refuses

    run(scenario())


# ---------------------------------------------------------------------------------------------- event hub
def test_hub_replays_latest_state_to_late_subscribers():
    async def scenario():
        hub = EventHub()
        hub.publish({"type": "status", "worker": "ready", "detail": ""})
        hub.publish({"type": "analyze.progress", "session": "s1", "phase": "decode", "pct": 10}, "s1")
        hub.publish({"type": "analyze.progress", "session": "s1", "phase": "depth", "pct": 50}, "s1")
        hub.publish({"type": "analyze.progress", "session": "s2", "phase": "depth", "pct": 5}, "s2")
        with hub.subscribe("s1") as sub:
            first, second = await sub.get(), await sub.get()
            assert first["type"] == "status"
            assert second["pct"] == 50  # only the latest progress is replayed
            assert sub.pending == 0
        hub.publish({"type": "ready", "session": "s1"}, "s1")
        with hub.subscribe("s1") as sub:
            assert [(await sub.get())["type"] for _ in range(2)] == ["status", "ready"]
        with hub.subscribe(None) as g:  # global: status only, no session state
            assert (await g.get())["type"] == "status" and g.pending == 0
        hub.forget("s1")
        with hub.subscribe("s1") as sub:
            assert sub.pending == 1

    run(scenario())


def test_hub_routing_between_sessions_and_global():
    async def scenario():
        hub = EventHub()
        with hub.subscribe("a") as a, hub.subscribe("b") as b, hub.subscribe(None) as g:
            hub.publish({"type": "ready", "session": "a"}, "a")
            hub.publish({"type": "status", "worker": "error", "detail": "x"})
            assert [(await a.get())["type"] for _ in range(2)] == ["ready", "status"]
            assert (await b.get())["type"] == "status" and b.pending == 0
            assert [(await g.get())["type"] for _ in range(2)] == ["ready", "status"]
        assert hub.subscribers == 0

    run(scenario())


def test_slow_subscriber_does_not_block_and_keeps_ready():
    async def scenario():
        hub = EventHub(max_queue=16)
        slow = hub.subscribe("s")  # never reads
        fast = hub.subscribe("s")
        got = []

        async def reader():
            async for e in fast:
                got.append(e)

        task = asyncio.ensure_future(reader())
        t = time.perf_counter()
        for i in range(2000):
            hub.publish({"type": "analyze.progress", "session": "s", "pct": i}, "s")
            if i % 100 == 0:
                await asyncio.sleep(0)  # let the fast reader run
        hub.publish({"type": "ready", "session": "s"}, "s")
        assert time.perf_counter() - t < 1.0  # publishing never waits for the slow client
        assert slow.pending <= 17 and slow.dropped > 1900
        items = [await slow.get() for _ in range(slow.pending)]
        assert items[-1]["type"] == "ready"  # never dropped
        assert [e["pct"] for e in items[:-1]] == sorted(e["pct"] for e in items[:-1])
        assert items[-2]["pct"] == 1999  # the newest progress survives, the oldest were dropped
        hub.close()
        await asyncio.wait_for(task, 5)
        assert got[-1]["type"] == "ready"

    run(scenario())


def test_hub_close_ends_iteration_after_draining():
    async def scenario():
        hub = EventHub()
        sub = hub.subscribe("s")
        hub.publish({"type": "ready", "session": "s"}, "s")
        hub.close()
        assert [e["type"] async for e in sub] == ["ready"]
        hub.publish({"type": "error", "session": "s"}, "s")  # ignored after close
        late = hub.subscribe("s")  # replays the remembered state, then ends
        assert [e["type"] async for e in late] == ["ready"]

    run(scenario())


# ------------------------------------------------------------------------------------------- websocket
def jpeg() -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    config = ServerConfig(port=PORT, data_dir=tmp_path, stub=True, web_dist=tmp_path / "nodist")
    app = create_app(config, on_created=lambda s: None)  # no automatic analysis in these tests
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.app_ = app
        c.portal.call(app.state.worker.wait_ready, 60)
        yield c


HOST = {"Host": f"127.0.0.1:{PORT}"}  # TestClient builds websocket URLs on "testserver"


def test_websocket_status_progress_ready_and_replay(client):
    app = client.app_
    sid = client.post("/api/sessions", files={"file": ("a.jpg", jpeg())}).json()["id"]

    async def fake_analyze():
        hub = app.state.hub

        def forward(event):
            if event.type == "progress":
                hub.publish(
                    {"type": "analyze.progress", "session": sid, "phase": "depth", "pct": event.data["pct"]},
                    sid,
                )

        handle = app.state.scheduler.submit(
            "echo_sleep",
            {"seconds": 0.2, "steps": 2},
            priority=Priority.ANALYZE,
            session=sid,
            on_event=forward,
        )
        out = await handle.result()
        hub.publish({"type": "ready", "session": sid}, sid)
        return out.status

    with client.websocket_connect(f"/api/sessions/{sid}/events", headers=HOST) as ws:
        first = ws.receive_json()
        assert first["type"] == "status" and first["worker"] == "ready"
        assert client.portal.call(fake_analyze) == "result"
        types = [ws.receive_json() for _ in range(3)]
        assert [m["type"] for m in types] == ["analyze.progress", "analyze.progress", "ready"]
        assert [m["pct"] for m in types[:2]] == [50.0, 100.0]

    # a page reload: late subscriber gets the status and the remembered state straight away
    with client.websocket_connect(f"/api/sessions/{sid}/events", headers=HOST) as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json() == {"type": "ready", "session": sid}

    with client.websocket_connect("/api/events", headers=HOST) as ws:  # global channel
        assert ws.receive_json()["type"] == "status"

    assert client.portal.call(lambda: app.state.hub.subscribers) == 0  # sockets unsubscribed on close


def test_websocket_unknown_session_refused_and_status_route(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/api/sessions/{'0' * 64}/events", headers=HOST):
            pass
    r = client.get("/api/status")
    body = r.json()
    assert r.status_code == 200 and body["worker"]["state"] == "ready"
    assert body["queue"]["length"] == 0 and set(body["timings"]) == {"interactive", "render", "analyze"}


def test_hub_close_on_shutdown_closes_websocket_cleanly(tmp_path):
    """The app's lifespan calls ``hub.close()`` first on shutdown; sockets must end with code 1001."""
    from starlette.websockets import WebSocketDisconnect

    config = ServerConfig(port=PORT, data_dir=tmp_path, stub=True, web_dist=tmp_path / "nodist")
    app = create_app(config)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.portal.call(app.state.worker.wait_ready, 60)
        with c.websocket_connect("/api/events", headers=HOST) as ws:
            assert ws.receive_json()["type"] == "status"
            c.portal.call(app.state.hub.close)
            with pytest.raises(WebSocketDisconnect) as info:
                ws.receive_json()
            assert info.value.code == 1001
        proc = app.state.worker.process
    assert proc is None or proc.exitcode is not None  # the worker was stopped with the app
