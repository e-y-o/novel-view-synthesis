"""End-to-end benchmark of the web server with the real depth model (M2 / T16).

    uv run python scripts/bench_server.py [--runs 5] [--soak-uploads 30] [--soak-renders 200] [--skip-soak]
                                          [--out data/bench]

Starts ``nvs serve`` (real model, port 8798, temporary data directory), waits for the worker, and measures over HTTP:

1. upload -> ``ready`` for 2 / 12 / 24 MP generated photos (plus the largest local sample photo if present), cold
   (new bytes), warm (the same bytes again: the preview is already on disk) and "cache cleared" (session deleted
   and the depth cache emptied, model still resident), median of ``--runs``, with the phase timeline taken from the
   session's WebSocket events;
2. coverage round trip (p50 / p95) and GPU / VRAM / CPU load (``nvidia-smi``, process CPU time) while coverage
   requests run back to back for 10 s;
3. full-resolution ``/render`` for 2 / 12 / 24 MP with the server's stage breakdown;
4. a soak: ``--soak-uploads`` different uploads and ``--soak-renders`` renders; VRAM before and after, worker
   process count, ``/api/status`` sanity.

Browser-side numbers (upload -> first frame, drag fps) come from ``web/e2e/bench.spec.ts``. Results are written to
``data/bench/bench-server-<timestamp>.json`` and printed as markdown tables.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PORT = 8798
BASE = f"http://127.0.0.1:{PORT}"
SIZES = {"2 MP": (1600, 1250), "12 MP": (4000, 3000), "24 MP": (6000, 4000)}


# ------------------------------------------------------------------------------------------------ helpers
def make_photo(width: int, height: int, seed: int) -> bytes:
    """A generated JPEG with structure (gradient, shapes, noise); distinct bytes per seed."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    img = np.stack([x / width * 200 + 30, y / height * 180 + 40, (x + y) / (width + height) * 220], -1)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, width - 200)), int(rng.integers(0, height - 200))
        w, h = int(rng.integers(60, 400)), int(rng.integers(60, 400))
        img[y0 : y0 + h, x0 : x0 + w] = rng.integers(20, 235, 3)
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def pose_body(yaw: float, pivot: float, **extra) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from nvs.core.camera import orbit_pose

    R, t = orbit_pose(yaw, 1.0, pivot)
    return {
        "R": R.tolist(),
        "t": t.tolist(),
        "orbit": {"yaw_deg": yaw, "pitch_deg": 1.0, "dolly": 0, "pan": [0, 0], "pivot_z": pivot},
        **extra,
    }


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def gpu_sample() -> tuple[float, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    mem, util = (float(v) for v in out.split(","))
    return mem, util


def spawn_children() -> list[int]:
    """Process ids of spawned worker processes (Windows: ``spawn_main`` in the command line)."""
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'spawn_main' } | ForEach-Object { $_.ProcessId }"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def cpu_seconds(pids: list[int]) -> float:
    if not pids:
        return 0.0
    ids = ",".join(str(p) for p in pids)
    cmd = f"(Get-Process -Id {ids} -ErrorAction SilentlyContinue | Measure-Object CPU -Sum).Sum"
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True
    ).stdout.strip()
    return float(out) if out else 0.0


class Server:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        env = dict(os.environ, NVS_ACCEPT_DEPTHPRO2_LICENSE="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "nvs.cli.main", "serve", "--port", str(PORT), "--data-dir", str(data_dir)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 180) -> float:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st = httpx.get(f"{BASE}/api/status", timeout=2).json()
                if st["worker"]["state"] == "ready":
                    return time.time() - t0
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError("worker did not become ready")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


async def phase_timeline(session_id: str, stop: asyncio.Event) -> list[tuple[str, float]]:
    """(phase, seconds since start) for every phase change on the session's event stream."""
    import websockets

    out: list[tuple[str, float]] = []
    t0 = time.perf_counter()
    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{PORT}/api/sessions/{session_id}/events",
            origin=BASE,
            additional_headers={"Host": f"127.0.0.1:{PORT}"},
        ) as ws:
            while not stop.is_set():
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 0.5))
                except TimeoutError:
                    continue
                if msg["type"] == "analyze.progress" and (not out or out[-1][0] != msg["phase"]):
                    out.append((msg["phase"], time.perf_counter() - t0))
                if msg["type"] in ("ready", "error"):
                    out.append((msg["type"], time.perf_counter() - t0))
                    break
    except Exception:
        pass
    return out


def upload_until_ready(client: httpx.Client, data: bytes, name: str) -> tuple[str, float, list]:
    """Seconds from the start of the upload to ready, and the phase timeline (best effort)."""
    sid = hashlib.sha256(data).hexdigest()
    t0 = time.perf_counter()
    r = client.post("/api/sessions", files={"file": (name, data)})
    r.raise_for_status()
    body = r.json()
    timeline: list[tuple[str, float]] = []
    if body["state"] != "ready":
        # attach to the event stream for the phase timeline while polling for the final state
        loop = asyncio.new_event_loop()
        stop = asyncio.Event()
        task = loop.create_task(phase_timeline(sid, stop))
        while True:
            s = client.get(f"/api/sessions/{sid}").json()
            if s["state"] in ("ready", "error"):
                assert s["state"] == "ready", s
                break
            loop.run_until_complete(asyncio.sleep(0.005))
        stop.set()
        timeline = loop.run_until_complete(task)
        loop.close()
        timeline = [(p, round(t - 0.0, 3)) for p, t in timeline]
    return sid, time.perf_counter() - t0, timeline


def stage_table(timeline: list[tuple[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for (phase, t), (_, nxt) in zip(
        timeline, timeline[1:] + [("end", timeline[-1][1] if timeline else 0)], strict=True
    ):
        out[phase] = round(out.get(phase, 0) + max(0.0, nxt - t), 3)
    return out


# ------------------------------------------------------------------------------------------------- benches
def bench_upload(client: httpx.Client, server: Server, runs: int, extra: dict[str, bytes]) -> dict:
    results: dict = {}
    sources = {k: None for k in SIZES} | {k: None for k in extra}
    for label in sources:
        cold, warm, cleared, stages = [], [], [], []
        for i in range(runs):
            data = extra[label] if label in extra else make_photo(*SIZES[label], seed=1000 + i)
            if (
                label in extra and i > 0
            ):  # a local photo has one set of bytes: later cold runs need the caches emptied
                shutil.rmtree(server.data_dir / "cache" / "depth", ignore_errors=True)
                sid = hashlib.sha256(data).hexdigest()
                client.delete(f"/api/sessions/{sid}")
            sid, t, timeline = upload_until_ready(client, data, f"{label}.jpg")
            cold.append(t)
            stages.append(stage_table(timeline))
            _, tw, _ = upload_until_ready(client, data, f"{label}.jpg")  # same bytes: preview already on disk
            warm.append(tw)
            client.delete(f"/api/sessions/{sid}")
            shutil.rmtree(
                server.data_dir / "cache" / "depth", ignore_errors=True
            )  # model resident, depth gone
            _, tc, _ = upload_until_ready(client, data, f"{label}.jpg")
            cleared.append(tc)
            client.delete(f"/api/sessions/{sid}")
        # a depth-cache hit: delete the session but keep the depth cache
        data = extra[label] if label in extra else make_photo(*SIZES[label], seed=7000)
        sid, _, _ = upload_until_ready(client, data, f"{label}.jpg")
        client.delete(f"/api/sessions/{sid}")
        cached = []
        for _ in range(runs):
            sid, tc, _ = upload_until_ready(client, data, f"{label}.jpg")
            cached.append(tc)
            client.delete(f"/api/sessions/{sid}")
        keys = sorted({k for s in stages for k in s})
        results[label] = {
            "cold_s": round(median(cold), 3),
            "warm_reupload_s": round(median(warm), 3),
            "cache_cleared_s": round(median(cleared), 3),
            "depth_cached_s": round(median(cached), 3),
            "cold_stages_s": {k: round(median([s.get(k, 0.0) for s in stages]), 3) for k in keys},
        }
        print(f"upload {label}: {results[label]}", flush=True)
    return results


def bench_interaction(client: httpx.Client, server: Server, seconds: float = 10.0) -> dict:
    data = make_photo(*SIZES["12 MP"], seed=4242)
    sid, _, _ = upload_until_ready(client, data, "interaction.jpg")
    meta = client.get(f"/api/sessions/{sid}").json()["meta"]
    pivot = meta["pivot_z"]
    times: list[float] = []
    mem, util = [], []
    pids = spawn_children() + [server.proc.pid]
    cpu0, t0 = cpu_seconds(pids), time.perf_counter()
    i = 0
    last_sample = 0.0
    while time.perf_counter() - t0 < seconds:
        yaw = 8 * np.sin(i * 0.07)
        t = time.perf_counter()
        r = client.post(f"/api/sessions/{sid}/coverage", json=pose_body(float(yaw), pivot))
        r.raise_for_status()
        times.append((time.perf_counter() - t) * 1000)
        if time.perf_counter() - last_sample > 0.25:
            m, u = gpu_sample()
            mem.append(m)
            util.append(u)
            last_sample = time.perf_counter()
        i += 1
    wall = time.perf_counter() - t0
    cpu = (cpu_seconds(pids) - cpu0) / wall * 100
    client.delete(f"/api/sessions/{sid}")
    out = {
        "requests": len(times),
        "per_second": round(len(times) / wall, 1),
        "coverage_ms_p50": round(pct(times, 0.5), 1),
        "coverage_ms_p95": round(pct(times, 0.95), 1),
        "gpu_util_pct_median": round(median(util), 1),
        "gpu_util_pct_max": max(util),
        "vram_mib_median": round(median(mem)),
        "vram_mib_max": max(mem),
        "server_cpu_pct_of_one_core": round(cpu),
    }
    print(f"interaction: {out}", flush=True)
    return out


def bench_render(client: httpx.Client, runs: int, extra: dict[str, bytes]) -> dict:
    out: dict = {}
    for label in list(SIZES) + list(extra):
        data = extra[label] if label in extra else make_photo(*SIZES[label], seed=5000)
        sid, _, _ = upload_until_ready(client, data, f"{label}.jpg")
        pivot = client.get(f"/api/sessions/{sid}").json()["meta"]["pivot_z"]
        rts, stages = [], []
        first = None
        for i in range(runs + 1):
            body = pose_body(
                3.0 + 0.5 * i, pivot, size="source", hole="alpha", fill_specks=True, format="png"
            )
            t = time.perf_counter()
            r = client.post(f"/api/sessions/{sid}/render", json=body).json()
            rt = time.perf_counter() - t
            if i == 0:
                first = rt  # the first render of a session decodes the source image
            else:
                rts.append(rt)
                stages.append(r["ms"])
        out[label] = {
            "first_render_s": round(first or 0, 3),
            "warm_round_trip_s": round(median(rts), 3),
            "stages_ms": {k: round(median([s[k] for s in stages]), 1) for k in stages[0]},
        }
        client.delete(f"/api/sessions/{sid}")
        print(f"render {label}: {out[label]}", flush=True)
    return out


def bench_soak(client: httpx.Client, server: Server, uploads: int, renders: int) -> dict:
    # warm up caches, then compare VRAM before and after
    sid, _, _ = upload_until_ready(client, make_photo(1600, 1250, seed=9000), "warm.jpg")
    pivot = client.get(f"/api/sessions/{sid}").json()["meta"]["pivot_z"]
    client.post(
        f"/api/sessions/{sid}/render", json=pose_body(2.0, pivot, size="source", hole="alpha", format="png")
    )
    vram0, _ = gpu_sample()
    children0 = spawn_children()
    t0 = time.perf_counter()
    for i in range(uploads):
        s2, _, _ = upload_until_ready(client, make_photo(1600, 1250, seed=9100 + i), f"soak{i}.jpg")
        client.delete(f"/api/sessions/{s2}")
    t_up = time.perf_counter() - t0
    vram_mid, _ = gpu_sample()
    t0 = time.perf_counter()
    for i in range(renders):
        r = client.post(
            f"/api/sessions/{sid}/render",
            json=pose_body(
                1.0 + i * 0.013, pivot, size="source", hole="alpha", format="png"
            ),  # every pose distinct
        )
        r.raise_for_status()
    t_r = time.perf_counter() - t0
    vram1, _ = gpu_sample()
    status = client.get("/api/status").json()
    children1 = spawn_children()
    out = {
        "uploads": uploads,
        "renders": renders,
        "uploads_total_s": round(t_up, 1),
        "renders_total_s": round(t_r, 1),
        "vram_mib_before": vram0,
        "vram_mib_after_uploads": vram_mid,
        "vram_mib_after": vram1,
        "vram_growth_mib": vram1 - vram0,
        "worker_processes_before": len(children0),
        "worker_processes_after": len(children1),
        "worker_state": status["worker"]["state"],
        "queue": status["queue"],
        "timings": {
            k: {kk: v[kk] for kk in ("count", "run_median_ms", "run_p95_ms")}
            for k, v in status["timings"].items()
        },
    }
    out["ok"] = (
        out["vram_growth_mib"] < 1024
        and out["worker_processes_after"] == out["worker_processes_before"] == 1
        and out["worker_state"] == "ready"
        and out["queue"]["length"] == 0
    )
    print(f"soak: {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--soak-uploads", type=int, default=30)
    p.add_argument("--soak-renders", type=int, default=200)
    p.add_argument("--skip-soak", action="store_true")
    p.add_argument("--soak-only", action="store_true", help="run only the soak")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "bench")
    args = p.parse_args(argv)

    extra: dict[str, bytes] = {}
    sample = ROOT / "data" / "samples" / "architecture_01.HEIC"
    if sample.is_file():
        extra["24 MP sample (HEIC)"] = sample.read_bytes()

    data_dir = Path(tempfile.mkdtemp(prefix="nvs-bench-"))
    server = Server(data_dir)
    result: dict = {"date": datetime.now().isoformat(timespec="seconds"), "runs": args.runs}
    try:
        result["worker_ready_s"] = round(server.wait_ready(), 1)
        print(f"worker ready after {result['worker_ready_s']} s; VRAM {gpu_sample()[0]:.0f} MiB", flush=True)
        with httpx.Client(base_url=BASE, timeout=300) as client:
            for _ in range(2):  # warm-up: CUDA kernels, allocator
                upload_until_ready(client, make_photo(1600, 1250, seed=1), "w.jpg")
            if not args.soak_only:
                result["upload"] = bench_upload(client, server, args.runs, extra)
                result["interaction"] = bench_interaction(client, server)
                result["render"] = bench_render(client, args.runs, extra)
            if not args.skip_soak:
                result["soak"] = bench_soak(client, server, args.soak_uploads, args.soak_renders)
    finally:
        server.stop()
        time.sleep(1)
        result["worker_processes_after_stop"] = len(spawn_children())
        shutil.rmtree(data_dir, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench-server-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {path}")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
