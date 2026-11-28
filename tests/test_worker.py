"""GPU worker process, IPC and stub backend (T03). No GPU needed: everything runs with ``stub=True``.

Each scenario starts a real spawned worker (Windows ``spawn`` semantics), so they are grouped to keep the
number of process starts low.
"""

import asyncio
import subprocess
import sys
import time

import pytest
import torch

from nvs.depth.base import DepthBackend, RawDepth
from nvs.depth.stub import StubDepthBackend
from nvs.worker.client import WorkerClient
from nvs.worker.protocol import READY, Request, WorkerConfig

STUB = WorkerConfig(stub=True)


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 90))


class started:
    """``async with started(...) as client``: a running stub worker that is always stopped afterwards."""

    def __init__(self, config=STUB, **kw):
        self.client = WorkerClient(config, **kw)

    async def __aenter__(self) -> WorkerClient:
        await self.client.start()
        await self.client.wait_ready(30)
        return self.client

    async def __aexit__(self, *exc):
        proc = self.client.process
        await self.client.stop()
        if proc is not None:
            assert proc.exitcode is not None, "worker process left running"


def req(rid, kind, **payload):
    return Request(id=rid, kind=kind, payload=payload)


def test_ready_ping_progress_cancel_unknown_job():
    async def scenario():
        t0 = time.perf_counter()
        client = WorkerClient(STUB)
        await client.start()
        status = await client.wait_ready(10)
        print(f"worker ready after {time.perf_counter() - t0:.2f} s: {status.detail}")
        assert status.state == READY

        # ping round trip
        await client.submit(req("warm", "ping")).wait(5)
        t = time.perf_counter()
        done = await client.submit(req("p1", "ping")).wait(5)
        rtt_ms = (time.perf_counter() - t) * 1000
        print(f"ping round trip {rtt_ms:.1f} ms")
        assert done.type == "result" and done.data["pong"] is True and done.data["stub"] is True
        assert rtt_ms < 50

        # echo_sleep: started, ordered progress, result
        handle = client.submit(req("e1", "echo_sleep", seconds=0.2, steps=4, echo={"k": [1, 2]}))
        events = [e async for e in handle]
        assert [e.type for e in events] == [
            "started",
            "progress",
            "progress",
            "progress",
            "progress",
            "result",
        ]
        assert [e.data["pct"] for e in events if e.type == "progress"] == [25.0, 50.0, 75.0, 100.0]
        assert events[-1].data == {"echo": {"k": [1, 2]}}

        # cancel a running job: ends with `cancelled` quickly
        handle = client.submit(req("c1", "echo_sleep", seconds=5.0, steps=50))
        await asyncio.sleep(0.2)
        t = time.perf_counter()
        client.cancel("c1")
        final = await handle.wait(5)
        assert final.type == "cancelled"
        assert time.perf_counter() - t < 1.0
        assert handle.events[0].type == "started"

        # cancel while still queued: never starts
        slow = client.submit(req("s1", "echo_sleep", seconds=0.5, steps=5))
        queued = client.submit(req("q1", "echo_sleep", seconds=5.0, steps=50))
        client.cancel("q1")
        assert (await queued.wait(5)).type == "cancelled"
        assert "started" not in [e.type for e in queued.events]
        assert (await slow.wait(5)).type == "result"

        # cancelling something already finished is harmless
        client.cancel("p1")

        # unknown kind: error, and the worker keeps running
        bad = await client.submit(req("u1", "no_such_job")).wait(5)
        assert bad.type == "error" and bad.data["code"] == "unknown_job"
        assert (await client.submit(req("p2", "ping")).wait(5)).type == "result"

        proc = client.process
        await client.stop()
        assert proc.exitcode == 0  # clean shutdown, no orphan

    run(scenario())


def test_kill_inflight_errors_then_restart_works():
    async def scenario():
        async with started(auto_restart=False) as client:
            handle = client.submit(req("k1", "echo_sleep", seconds=10.0, steps=100))
            await asyncio.sleep(0.3)
            old = client.process
            old.kill()
            final = await handle.wait(10)
            assert final.type == "error" and final.data["code"] == "worker_died"
            assert client.status.state == "error"

            again = await client.submit(req("k2", "ping")).wait(5)  # no worker: fails fast, no hang
            assert again.type == "error" and again.data["code"] == "worker_died"

            await client.restart()
            await client.wait_ready(30)
            assert client.process is not old
            done = await client.submit(req("k3", "ping")).wait(5)
            assert done.type == "result"
        assert old.exitcode is not None

    run(scenario())


def test_auto_restart_with_backoff_then_gives_up():
    async def scenario():
        async with started(backoff=(0.1, 0.1)) as client:
            for attempt in range(2):  # two crashes are survived
                client.process.kill()
                await asyncio.sleep(0.3)  # let the death be noticed
                await client.wait_ready(30)
                assert (await client.submit(req(f"a{attempt}", "ping")).wait(5)).type == "result"
            client.process.kill()  # third consecutive failure: out of restarts
            for _ in range(100):
                if "giving up" in client.status.detail:
                    break
                await asyncio.sleep(0.1)
            assert client.status.state == "error" and "giving up" in client.status.detail
            assert not client.alive

    run(scenario())


def test_status_listener_and_no_preload():
    async def scenario():
        seen = []
        cfg = WorkerConfig(stub=True, preload_depth=False)
        async with started(cfg) as client:
            client.add_status_listener(seen.append)
            assert client.status.state == READY and "not preloaded" in client.status.detail
            await client.restart()
            await client.wait_ready(30)
            states = [s.state for s in seen]
            assert "starting" in states and states[-1] == READY

    run(scenario())


def test_stop_without_start_and_double_stop():
    async def scenario():
        client = WorkerClient(STUB)
        await client.stop()
        await client.start()
        await client.wait_ready(30)
        await client.stop()
        await client.stop()

    run(scenario())


def test_api_app_does_not_import_torch_via_worker_client():
    code = (
        "import sys\nimport nvs.api.app, nvs.worker.client, nvs.worker.protocol\n"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stderr


# ------------------------------------------------------------------------------------------ stub backend
def test_stub_backend_shape_finite_deterministic():
    backend = StubDepthBackend()
    assert isinstance(backend, DepthBackend)
    rgb = torch.rand(3, 90, 120, generator=torch.Generator().manual_seed(0))
    a = backend.infer(rgb, None)
    b = backend.infer(torch.rand(3, 90, 120), None)  # different pixels: same scene
    assert isinstance(a, RawDepth)
    assert a.depth.shape == (90, 120) and a.depth.dtype == torch.float32
    assert torch.isfinite(a.depth).all()
    assert torch.equal(a.depth, b.depth)
    assert a.infer_size == (90, 120) and a.scale == 1.0
    assert a.f_px == pytest.approx(0.9 * 120)
    assert backend.estimate_focal(rgb) == pytest.approx(0.9 * 120)
    assert backend.infer(rgb, 77.0).f_px == 77.0


def test_stub_scene_structure():
    d = StubDepthBackend().infer(torch.zeros(3, 100, 100), None).depth
    assert d[0, 0] > d[-1, 0]  # far at the top, near at the bottom
    assert d[50, 50] == pytest.approx(1.5)  # ellipse in the middle
    assert d[50, 50] < d[-1, 50]  # ellipse in front of the nearest ground
    with pytest.raises(ValueError):
        StubDepthBackend().infer(torch.zeros(4, 10, 10), None)
